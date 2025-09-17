import sys, os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger

from src.config_entities.config_entity import ModelTrainingConfig
from src.config_entities.artifact_entity import DataProcessingArtifact, ModelTrainingArtifact

from src.utils.common import (
    detect_timestamp_column,
    save_object,
    save_numpy_array_data,
    load_object,
    load_numpy_array_data,
    save_json,
    per_window_score,
    per_timestep_errors,
    pick_ensemble_threshold,
    _tensor_to_numpy,
    _ensure_dir_for,
    _cfg_path
)
from src.model_components.LSTM_ARCHITECTURE.lstm_AE import LSTMAutoencoder
from src.model_components.GRU_ARCHITECTURE.gru_AE import GRUAutoencoder
from src.model_components.CNN1D_ARCHITECTURE.cnn_1d_AE import CNNAutoencoder1D


try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception as _e:
    mlflow = None  # type: ignore
    _MLFLOW_AVAILABLE = False
    


class ModelTraining:

    def __init__(self, data_processing_artifact: DataProcessingArtifact, model_training_config: ModelTrainingConfig):
        try:
            self.data_processing_artifact = data_processing_artifact
            self.model_training_config = model_training_config

            # MLflow wiring 
            self.mlflow_enabled: bool = bool(getattr(self.model_training_config, "use_mlflow", True)) and _MLFLOW_AVAILABLE
            if self.mlflow_enabled:
                try:
                    tracking_uri = getattr(self.model_training_config, "mlflow_tracking_uri", None)
                    if tracking_uri:
                        mlflow.set_tracking_uri(tracking_uri)
                    exp_name = getattr(self.model_training_config, "mlflow_experiment_name", "anomaly_detection")
                    mlflow.set_experiment(exp_name)
                    logger.logging.info(f"MLflow enabled. Experiment='{exp_name}', URI='{tracking_uri}'")
                except Exception as e:
                    logger.logging.info(f"Failed to initialize MLflow; disabling. Reason: {e}")
                    self.mlflow_enabled = False
        except Exception as e:
            raise AnomalyDetectionException(e, sys)

   
    # DataLoaders 
    def convert_dataset_to_dataloader(self):
        """Create DataLoaders from datasets stored in the data_processing_artifact and persist them.
        Assumes the artifact already holds dataset objects (not paths)."""
        try:
            # load actual dataset objects instead of using raw paths
            train_dataset = load_object(self.data_processing_artifact.train_dataset_path)
            val_dataset   = load_object(self.data_processing_artifact.val_dataset_path)
            test_dataset  = load_object(self.data_processing_artifact.test_dataset_path)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.model_training_config.batch_size,
                shuffle=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.model_training_config.batch_size,
                shuffle=False,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.model_training_config.batch_size,
                shuffle=False,
            )

            save_object(
                file_path=self.model_training_config.train_dataloader_path,
                obj=train_loader,
            )
            save_object(
                file_path=self.model_training_config.val_dataloader_path,
                obj=val_loader,
            )
            save_object(
                file_path=self.model_training_config.test_dataloader_path,
                obj=test_loader,
            )

            logger.logging.info("Dataloader successfully saved")

            return train_loader, val_loader, test_loader
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    
    # Core Training of One Model
    def train_one_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        lr: float,
        weight_decay: float,
        device: torch.device,
        model_name: str = "model",
        grad_clip: Optional[float] = None,
        use_amp: bool = False,
    ) -> Dict[str, Any]:
        """
        Train a single autoencoder model and save its BEST weights (lowest val loss).

        Returns:
            dict with {'model': best_model, 'history': {...}, 'best_val': float, 'weights_path': str}
        """
        try:
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.MSELoss(reduction="mean")
            scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda")

            best_val = float("inf")
            best_wts = None
            history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

            
            run_ctx = (
                mlflow.start_run(run_name=f"{model_name}_AE", nested=True) if self.mlflow_enabled else None
            )
            if run_ctx:
                run_ctx.__enter__()
                # Log basic hyperparams
                mlflow.log_params(
                    {
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "num_epochs": num_epochs,
                        "grad_clip": grad_clip if grad_clip is not None else "none",
                        "use_amp": use_amp,
                        "device": str(device),
                        "model_name": model_name,
                    }
                )

            for ep in range(1, num_epochs + 1):
                # ----- train -----
                model.train()
                total_loss = 0.0
                total_seen = 0

                for batch in train_loader:
                    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                    xb = xb.to(device).float()

                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        yhat = model(xb)
                        loss = criterion(yhat, xb)

                    scaler.scale(loss).backward()
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item() * xb.size(0)
                    total_seen += xb.size(0)

                train_loss = total_loss / max(1, total_seen)

                # ----- validate -----
                model.eval()
                val_loss_sum = 0.0
                val_seen = 0
                with torch.no_grad():
                    for batch in val_loader:
                        xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                        xb = xb.to(device).float()
                        yhat = model(xb)
                        loss = criterion(yhat, xb)
                        val_loss_sum += loss.item() * xb.size(0)
                        val_seen += xb.size(0)
                val_loss = val_loss_sum / max(1, val_seen)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                print(f"[{model_name}] Epoch {ep}/{num_epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

                if self.mlflow_enabled:
                    mlflow.log_metric(f"{model_name}_train_loss", train_loss, step=ep)
                    mlflow.log_metric(f"{model_name}_val_loss", val_loss, step=ep)

                # keep best
                if val_loss < best_val:
                    best_val = val_loss
                    best_wts = model.state_dict()

            # restore best weights
            if best_wts is not None:
                model.load_state_dict(best_wts)

            # save final best model weights
            weights_path = os.path.join(self.model_training_config.model_training_dir, f"{model_name}_autoencoder.pt")
            _ensure_dir_for(weights_path)
            torch.save(model.state_dict(), weights_path)

            if self.mlflow_enabled:
                mlflow.log_metric(f"{model_name}_best_val_loss", best_val)
                if os.path.exists(weights_path):
                    mlflow.log_artifact(weights_path, artifact_path=f"weights/{model_name}")

            # Close child run if any
            if run_ctx:
                run_ctx.__exit__(None, None, None)

            return {"model": model, "history": history, "best_val": best_val, "weights_path": weights_path}

        except Exception as e:
            if self.mlflow_enabled and mlflow.active_run() is not None:
                try:
                    mlflow.end_run(status="FAILED")
                except Exception:
                    pass
            raise AnomalyDetectionException(e, sys) from e

    # Inference utilities
    @torch.no_grad()
    def reconstruct_all(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Reconstruct **all** batches in a loader and return:
            preds: (N, L, C)
            trues: (N, L, C)
            labels: (N,) or None
            timestamps: (N,) or None
        """
        try:
            device = device or getattr(self.model_training_config, "device", torch.device("cpu"))
            model = model.to(device).eval()
            preds: List[np.ndarray] = []
            trues: List[np.ndarray] = []
            labs: List[Optional[np.ndarray]] = []
            times: List[Optional[np.ndarray]] = []

            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    xb = batch[0]
                    lab = batch[1] if len(batch) > 1 else None
                    ts = batch[2] if len(batch) > 2 else None
                else:
                    xb, lab, ts = batch, None, None

                xb = xb.to(device).float()
                yhat = model(xb)

                preds.append(yhat.detach().cpu().numpy())
                trues.append(xb.detach().cpu().numpy())
                labs.append(_tensor_to_numpy(lab))
                times.append(_tensor_to_numpy(ts))

            preds_arr = np.concatenate(preds, axis=0)
            trues_arr = np.concatenate(trues, axis=0)

            labels = None
            if any(l is not None for l in labs):
                labels = np.concatenate([l for l in labs if l is not None], axis=0)

            timestamps = None
            if any(t is not None for t in times):
                timestamps = np.concatenate([t for t in times if t is not None], axis=0)

            return preds_arr, trues_arr, labels, timestamps

        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e


    # Ensemble utilities
    def compute_ensemble_from_model_errors(
        self,
        errs_by_model: Dict[str, np.ndarray],
        normalization: str = "zscore",
        normalizers: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        model_names = sorted(errs_by_model.keys())
        M = len(model_names)
        if M == 0:
            raise ValueError("errs_by_model is empty")
        N = errs_by_model[model_names[0]].shape[0]
        mat = np.vstack([errs_by_model[m] for m in model_names])  # (M, N)

        if normalization == "none":
            ens = mat.mean(axis=0)
            return ens, {m: {"mean": 0.0, "std": 1.0} for m in model_names}
        elif normalization == "zscore":
            if normalizers is None:
                normalizers = {}
                for m in model_names:
                    mu = float(errs_by_model[m].mean())
                    sd = float(errs_by_model[m].std() + 1e-12)
                    normalizers[m] = {"mean": mu, "std": sd}
            # Apply z-score per model using provided/learned normalizers
            z = []
            for m in model_names:
                mu = normalizers[m]["mean"]
                sd = normalizers[m]["std"]
                z.append((errs_by_model[m] - mu) / (sd if sd > 0 else 1.0))
            z = np.vstack(z)  # (M, N)
            ens = z.mean(axis=0)
            return ens, normalizers
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

    # Full training + ensemble build
    def training_and_build_ensemble(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        lr: float,
    ) -> Dict[str, Any]:
        """
        Train LSTM/GRU/CNN AEs, compute per-model errors, build ensemble error,
        learn a single ensemble threshold on VALIDATION, and flag TEST with timestamps.
        Also logs metrics & artifacts to MLflow when enabled.
        """
        try:
            # Discover shape from first batch
            for batch in train_loader:
                x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
                seq_len = int(x0.shape[1])
                n_features = int(x0.shape[2])
                break
            print(f"Inferred data shape: seq_len={seq_len}, n_features={n_features}")

            models: Dict[str, nn.Module] = {
                "lstm": LSTMAutoencoder(
                    seq_len=seq_len,
                    n_features=n_features,
                    hidden_size=64,
                    latent_size=32,
                    num_layers=1,
                    dropout=0.0,
                ),
                "gru": GRUAutoencoder(
                    input_dim=n_features,
                    hidden_dim=64,
                    latent_dim=32,
                    num_layers=1,
                    dropout=0.0,
                    bidirectional=False,
                ),
                "cnn": CNNAutoencoder1D(
                    seq_len=seq_len,
                    in_channels=n_features,
                    channels=[32, 64, 128],
                    kernel_sizes=[3, 3, 3],
                    dilations=[1, 2, 4],
                    use_residual=True,
                    norm="batch",
                    act="relu",
                    dropout=0.1,
                    up_mode="transpose",
                    latent_dim=64,
                ),
            }

            model_config = {
                "lstm": {
                    "seq_len": seq_len,
                    "n_features": n_features,
                    "hidden_size": 64,
                    "latent_size": 32,
                    "num_layers": 1,
                    "dropout": 0.0,
                },
                "gru": {
                    "input_dim": n_features,
                    "hidden_dim": 64,
                    "latent_dim": 32,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "bidirectional": False,
                },
                "cnn": {
                    "seq_len": seq_len,
                    "in_channels": n_features,
                    "channels": [32, 64, 128],
                    "kernel_sizes": [3, 3, 3],
                    "dilations": [1, 2, 4],
                    "use_residual": True,
                    "norm": "batch",
                    "act": "relu",
                    "dropout": 0.1,
                    "up_mode": "transpose",
                    "latent_dim": 64,
                },
            }

            # Save model config for inference phase
            _ensure_dir_for(self.model_training_config.model_config_path)
            save_json(file_path=self.model_training_config.model_config_path, obj=model_config)
            logger.logging.info(f"Model config saved at: {self.model_training_config.model_config_path}")

            # Parent MLflow run
            parent_ctx = mlflow.start_run(run_name="ensemble_training") if self.mlflow_enabled else None
            if parent_ctx:
                parent_ctx.__enter__()
                mlflow.log_params(
                    {
                        "batch_size": self.model_training_config.batch_size,
                        "num_epochs": num_epochs,
                        "lr": lr,
                        "grad_clip": getattr(self.model_training_config, "grad_clip", None),
                        "use_amp": getattr(self.model_training_config, "use_amp", False),
                        "ensemble_normalization": getattr(self.model_training_config, "ensemble_normalization", "zscore"),
                        "threshold_method": getattr(self.model_training_config, "threshold_method", "quantile"),
                        "threshold_q": getattr(self.model_training_config, "threshold_q", None),
                        "threshold_k": getattr(self.model_training_config, "threshold_k", None),
                    }
                )

            trained: Dict[str, Any] = {}
            histories: Dict[str, Any] = {}

            for name, model in models.items():
                print(f"\n=== Training {name.upper()} AE ===")
                out = self.train_one_model(
                    model,
                    train_loader,
                    val_loader,
                    num_epochs=num_epochs,
                    lr=lr,
                    weight_decay=getattr(self.model_training_config, "weight_decay", 0.0),
                    device=self.model_training_config.device,
                    model_name=name,
                    grad_clip=getattr(self.model_training_config, "grad_clip", None),
                    use_amp=getattr(self.model_training_config, "use_amp", False),
                )
                trained[name] = out["model"]
                histories[name] = out["history"]

            results: Dict[str, Any] = {"models": trained, "histories": histories}

            # ---- Validation errors per model ----
            val_errs_by_model: Dict[str, np.ndarray] = {}
            for name, model in trained.items():
                preds, trues, _, _ = self.reconstruct_all(model, val_loader, device=self.model_training_config.device)
                errs = per_window_score(preds, trues)
                val_errs_by_model[name] = errs

                # Save & log each model's validation reconstruction errors
                val_err_path = os.path.join(self.model_training_config.model_training_dir, f"val_errors_{name}.npy")
                _ensure_dir_for(val_err_path)
                np.save(val_err_path, errs)
                if self.mlflow_enabled:
                    mlflow.log_metric(f"{name}_val_err_mean", float(np.mean(errs)))
                    mlflow.log_metric(f"{name}_val_err_p95", float(np.percentile(errs, 95)))
                    mlflow.log_artifact(val_err_path, artifact_path=f"errors/val/{name}")

            # ---- Ensemble on VALIDATION ----
            normalization = getattr(self.model_training_config, "ensemble_normalization", "zscore")
            val_ensemble_scores, normalizers = self.compute_ensemble_from_model_errors(
                val_errs_by_model, normalization=normalization, normalizers=None
            )
            val_ens_path = _cfg_path(self.model_training_config, "val_error_ensemble_path", "val_ensemble_scores.npy")
            _ensure_dir_for(val_ens_path)
            np.save(val_ens_path, val_ensemble_scores)
            if self.mlflow_enabled:
                mlflow.log_artifact(val_ens_path, artifact_path="ensemble/validation")

            # ---- Threshold from VALIDATION ----
            thr = pick_ensemble_threshold(
                val_ensemble_scores,
                method=getattr(self.model_training_config, "threshold_method", "quantile"),
                q=getattr(self.model_training_config, "threshold_q", 0.99),
                k=getattr(self.model_training_config, "threshold_k", 3.0),
            )
            print(
                f"\n[Ensemble] normalization={normalization} | method={getattr(self.model_training_config, 'threshold_method', 'quantile')} | threshold={thr:.6f}"
            )
            if self.mlflow_enabled:
                mlflow.log_metric("ensemble_threshold", float(thr))

            ensemble_meta = {
                "normalization": normalization,
                "normalizers": normalizers,
                "threshold": float(thr),
                "threshold_method": getattr(self.model_training_config, "threshold_method", "quantile"),
                "threshold_params": {
                    "q": getattr(self.model_training_config, "threshold_q", None),
                    "k": getattr(self.model_training_config, "threshold_k", None),
                },
            }

            # ensuring JSON-serializable
            ensemble_meta["normalizers"] = {
                k: {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in ensemble_meta["normalizers"].items()
            }
            ensemble_meta_path = _cfg_path(self.model_training_config, "ensemble_meta_path", "ensemble_meta.json")
            _ensure_dir_for(ensemble_meta_path)
            save_json(ensemble_meta, ensemble_meta_path)
            if self.mlflow_enabled and os.path.exists(ensemble_meta_path):
                mlflow.log_artifact(ensemble_meta_path, artifact_path="ensemble/meta")

            # ---- TEST: per-model errors, ensemble scores, flags ----
            test_errs_by_model: Dict[str, np.ndarray] = {}
            test_timestamps: Optional[np.ndarray] = None

            for name, model in trained.items():
                preds, trues, _, timestamps = self.reconstruct_all(model, test_loader, device=self.model_training_config.device)
                errs = per_window_score(preds, trues)
                test_errs_by_model[name] = errs

                test_err_path = os.path.join(self.model_training_config.model_training_dir, f"test_errors_{name}.npy")
                _ensure_dir_for(test_err_path)
                np.save(test_err_path, errs)
                if self.mlflow_enabled:
                    mlflow.log_metric(f"{name}_test_err_mean", float(np.mean(errs)))
                    mlflow.log_metric(f"{name}_test_err_p95", float(np.percentile(errs, 95)))
                    mlflow.log_artifact(test_err_path, artifact_path=f"errors/test/{name}")

                if test_timestamps is None:
                    test_timestamps = timestamps

            # Build ensemble TEST scores using SAME normalizers learned on VALIDATION
            test_ensemble_scores, _ = self.compute_ensemble_from_model_errors(
                test_errs_by_model, normalization=normalization, normalizers=normalizers
            )
            test_ens_path = _cfg_path(self.model_training_config, "test_error_ensemble_path", "test_ensemble_scores.npy")
            _ensure_dir_for(test_ens_path)
            np.save(test_ens_path, test_ensemble_scores)
            if self.mlflow_enabled:
                mlflow.log_artifact(test_ens_path, artifact_path="ensemble/test")

            test_flags = (test_ensemble_scores > thr).astype(int)
            flags_path = _cfg_path(self.model_training_config, "test_flags_ensemble_path", "test_flags.npy")
            _ensure_dir_for(flags_path)
            np.save(flags_path, test_flags)
            if self.mlflow_enabled:
                mlflow.log_artifact(flags_path, artifact_path="ensemble/test")

            # Timestamp-aware JSON results
            results_table: List[Dict[str, Any]] = []
            N = test_ensemble_scores.shape[0]
            for i in range(N):
                ts = (
                    str(test_timestamps[i])
                    if (test_timestamps is not None and i < len(test_timestamps))
                    else str(i)
                )
                results_table.append(
                    {
                        "timestamp": ts,
                        "ensemble_error": float(test_ensemble_scores[i]),
                        "flag": int(test_flags[i]),
                    }
                )
            test_results_json_path = _cfg_path(
                self.model_training_config, "ensemble_test_results_path", "ensemble_test_results.json"
            )
            _ensure_dir_for(test_results_json_path)
            save_json(results_table, test_results_json_path)
            print(f"Saved timestamped results -> {test_results_json_path}")
            if self.mlflow_enabled and os.path.exists(test_results_json_path):
                mlflow.log_artifact(test_results_json_path, artifact_path="ensemble/test")

            # Close parent run
            if parent_ctx:
                parent_ctx.__exit__(None, None, None)

            # return object
            results["ensemble"] = ensemble_meta
            results["val"] = {"per_model_errors": val_errs_by_model, "ensemble_errors": val_ensemble_scores}
            results["test"] = {
                "per_model_errors": test_errs_by_model,
                "ensemble_errors": test_ensemble_scores,
                "flags": test_flags,
                "timestamps": test_timestamps,
            }
            return results
        except Exception as e:
            if self.mlflow_enabled and mlflow.active_run() is not None:
                try:
                    mlflow.end_run(status="FAILED")
                except Exception:
                    pass
            raise AnomalyDetectionException(e, sys) from e

    # Orchestration entrypoint
    def initiate_training_and_build_ensemble(self) -> ModelTrainingArtifact:
        try:
            # First, ensure dataloaders are created & persisted
            train_loader, val_loader, test_loader = self.convert_dataset_to_dataloader()

            # (Optional safety check: if you want to always reload from disk)
            # train_loader = load_object(self.model_training_config.train_dataloader_path)
            # val_loader   = load_object(self.model_training_config.val_dataloader_path)
            # test_loader  = load_object(self.model_training_config.test_dataloader_path)

            results = self.training_and_build_ensemble(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_epochs=self.model_training_config.num_epoch,
                lr=self.model_training_config.learning_rate,
            )

            
            # Build artifact with paths consistent with what was saved
            artifact = ModelTrainingArtifact(
                lstm_weights_path=os.path.join(
                    self.model_training_config.model_training_dir, "lstm_autoencoder.pt"
                ),
                gru_weights_path=os.path.join(
                    self.model_training_config.model_training_dir, "gru_autoencoder.pt"
                ),
                cnn_weights_path=os.path.join(
                    self.model_training_config.model_training_dir, "cnn_autoencoder.pt"
                ),
                ensemble_meta_path=_cfg_path(
                    self.model_training_config,
                    "ensemble_meta_path",
                    self.model_training_config.ensemble_meta_path,
                ),
                scaler_pkl_path=self.data_processing_artifact.preprocessor,
                model_config_path=self.model_training_config.model_config_path,
            )
            return artifact

        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e
