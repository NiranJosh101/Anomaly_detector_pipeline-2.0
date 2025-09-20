import os
import sys
import io
import json
import pickle
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.utils.common import (
    load_json,
    mse_per_window,
    sliding_windows,
    window_right_edge_timestamps,
    batched_reconstruction_errors,
    compute_ensemble_from_model_errors
)

from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger


from src.config_entities.config_entity import ModelTrainingConfig
from src.config_entities.artifact_entity import ModelTrainingArtifact, DataProcessingArtifact, ModelInferenceArtifact

from model_components.LSTM_ARCHITECTURE.lstm_AE import LSTMAutoencoder
from model_components.GRU_ARCHITECTURE.gru_AE import GRUAutoencoder
from model_components.CNN1D_ARCHITECTURE.cnn_1d_AE import CNNAutoencoder1D




class AEEnsemblePredictor:
    def __init__(self, data_processing_artifact: DataProcessingArtifact, model_training_artifact: ModelTrainingArtifact, model_training_config: ModelTrainingConfig):
        try:
            self.model_training_config = model_training_config
            self.model_training_artifact = model_training_artifact
            self.data_processing_artifact = data_processing_artifact
            self.device = model_training_config.device
            self.batch_size = model_training_config.batch_size

            # Load window config
            cfg_path = self.data_processing_artifact.window_config_path  
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"Missing window config file at {cfg_path}")
            cfg = load_json(cfg_path)
            self.W: int = int(cfg["window_size"])
            self.step: int = int(cfg.get("step_size", 1))
            self.feature_order: List[str] = list(cfg["feature_order"])
            self.time_col: str = cfg.get("timestamp_col", "time")

            n_features = len(self.feature_order)

            # Optional scaler
            scaler_path = os.path.join(self.data_processing_artifact.preprocessor)
            self.scaler = None
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            # Ensemble meta
            meta_path = os.path.join(self.model_training_artifact.ensemble_meta_path)
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Missing ensemble meta file at {meta_path}")
            meta = load_json(meta_path)
            self.normalization: str = meta.get("normalization", "none")
            self.normalizers: Dict[str, Dict[str, float]] = meta.get("normalizers", {})
            self.threshold: float = float(meta["threshold"])

            # Model hyperparams
            model_cfg_path = os.path.join(self.model_training_artifact.model_config_path)
            if os.path.exists(model_cfg_path):
                mc = load_json(model_cfg_path)
            else:
                mc = {
                    "lstm": {"hidden_size": 64, "latent_size": 32, "num_layers": 1, "dropout": 0.0, "bidirectional": False},
                    "gru":  {"hidden_dim": 64, "latent_dim": 32, "num_layers": 1, "dropout": 0.0, "bidirectional": False},
                    "cnn":  {"channels": [32,64,128], "kernel_sizes": [3,3,3], "dilations": [1,2,4],
                            "use_residual": True, "norm": "batch", "act": "relu", "dropout": 0.1,
                            "up_mode": "transpose", "latent_dim": 64}
                }

            # Instantiate models
            self.models: Dict[str, nn.Module] = {}

            lstm = LSTMAutoencoder(
                seq_len=self.W, n_features=n_features,
                hidden_size=mc["lstm"]["hidden_size"], latent_size=mc["lstm"]["latent_size"],
                num_layers=mc["lstm"]["num_layers"], dropout=mc["lstm"]["dropout"],
                bidirectional=mc["lstm"].get("bidirectional", False)
            )
            lstm.load_state_dict(torch.load(os.path.join(self.model_training_config.model_training_dir, "lstm_autoencoder.pt"), map_location=self.model_training_config.device))
            lstm.eval().to(self.model_training_config.device)
            self.models["lstm"] = lstm

            gru = GRUAutoencoder(
                input_dim=n_features, hidden_dim=mc["gru"]["hidden_dim"],
                latent_dim=mc["gru"]["latent_dim"], num_layers=mc["gru"]["num_layers"],
                dropout=mc["gru"]["dropout"], bidirectional=mc["gru"].get("bidirectional", False)
            )
            gru.load_state_dict(torch.load(os.path.join(self.model_training_config.model_training_dir, "gru_autoencoder.pt"), map_location=self.model_training_config.device))
            gru.eval().to(self.model_training_config.device)
            self.models["gru"] = gru

            cnn = CNNAutoencoder1D(
                seq_len=self.W, in_channels=n_features,
                channels=mc["cnn"]["channels"],
                kernel_sizes=mc["cnn"]["kernel_sizes"],
                dilations=mc["cnn"]["dilations"],
                use_residual=mc["cnn"].get("use_residual", True),   
                norm=mc["cnn"].get("norm", "batch"),                
                act=mc["cnn"].get("act", "relu"),                   
                dropout=mc["cnn"].get("dropout", 0.0),              
                up_mode=mc["cnn"].get("up_mode", "nearest"),        
                latent_dim=mc["cnn"]["latent_dim"]
            )
            cnn.load_state_dict(torch.load(os.path.join(self.model_training_config.model_training_dir, "cnn_autoencoder.pt"), map_location=self.model_training_config.device))
            cnn.eval().to(self.model_training_config.device)
            self.models["cnn"] = cnn

        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        

    def predict_from_dataframe(self, df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Ensure df is a pandas DataFrame
            if not isinstance(df, pd.DataFrame):
                expected_cols = ([self.time_col] + self.feature_order) if self.time_col else list(self.feature_order)
                try:
                    df = pd.DataFrame(df, columns=expected_cols)
                except Exception as e:
                    raise ValueError(
                        f"Input must be a DataFrame with columns {expected_cols}, got {type(df)} instead: {e}"
                    )

            # Validate feature columns are present
            missing = [c for c in self.feature_order if c not in df.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")

            
            if self.time_col and self.time_col in df.columns:
                ts = df[self.time_col].astype(str).to_numpy()
            else:
                ts = np.array([str(i) for i in range(len(df))])

            # Extract features
            X_df = df[self.feature_order].astype(np.float32)

            # Scale/transform
            if self.scaler is not None:
                try:
                    X_scaled = self.scaler.transform(X_df)
                except Exception as e:
                    raise ValueError(
                        "Scaler/Preprocessor rejected the input. "
                        f"feature_order={self.feature_order}. Original error: {e}"
                    )
            else:
                X_scaled = X_df.values

            # Work with NumPy array
            X = np.asarray(X_scaled, dtype=np.float32)

            # Create sliding windows 
            Xw = sliding_windows(X, self.W, self.step)
            Tw = window_right_edge_timestamps(ts, self.W, self.step)

            if Xw.shape[0] == 0:
                raise HTTPException(status_code=400, detail="Input sequence too short for configured window size.")

            # Batched inference 
            per_model_errs: Dict[str, np.ndarray] = {}
            with torch.no_grad():
                for name, model in self.models.items():
                    per_model_errs[name] = batched_reconstruction_errors(
                        model, Xw, self.device, batch_size=self.batch_size
                    )

            ens_scores = compute_ensemble_from_model_errors(
                per_model_errs, normalization=self.normalization, normalizers=self.normalizers
            )
            flags = (ens_scores > self.threshold).astype(int)

            results = []
            for i in range(len(ens_scores)):
                results.append({
                    "timestamp": str(Tw[i]) if Tw is not None and i < len(Tw) else str(i),
                    "score": float(ens_scores[i]),
                    "flag": int(flags[i]),
                })

            output_dict = {"predictions": results}

            # compute metrics in same schema as training 
            feature_means = np.mean(X, axis=0).tolist()
            feature_stds = np.std(X, axis=0).tolist()
            global_mean = float(np.mean(X))
            global_std = float(np.std(X))

            error_median = float(np.median(ens_scores))
            error_p95 = float(np.percentile(ens_scores, 95))

            anomaly_rate = float(np.mean(flags))

            live_metrics = {
                "feature_stats": {
                    "mean": feature_means,
                    "std": feature_stds,
                    "global_mean": global_mean,
                    "global_std": global_std,
                },
                "error_stats": {
                    "median": error_median,
                    "p95": error_p95,
                },
                "anomaly_rate": anomaly_rate,
            }
            

            # Save predictions 
            save_path = save_path or self.model_training_config.inference_results_file
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(output_dict, f, indent=4)

            # Save metrics
            metrics_path = self.model_training_config.model_inference_metrics_file
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(live_metrics, f, indent=4)

            return output_dict

        except Exception as e:
            raise AnomalyDetectionException(e, sys)



    def initiate_model_prediction(self, df: pd.DataFrame) -> str:
        try:
            results = self.predict_from_dataframe(df)

            artifact = ModelInferenceArtifact(
                results_path=self.model_training_config.inference_results_file
            )
        except Exception as e:
            raise AnomalyDetectionException(e, sys)
