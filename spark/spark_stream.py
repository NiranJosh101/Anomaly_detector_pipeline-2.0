# spark_stream.py
"""
Spark Structured Streaming job with inference using batch-style AEEnsemblePredictor class.

Design notes:
- We create SparkAnomalyStreamer which:
  - Loads window config, scaler, and AEEnsemblePredictor identical to your batch loader.
  - Starts Spark Structured Streaming reading from RAW_TOPIC.
  - Uses foreachBatch to call self._process_microbatch(...) on each micro-batch.
  - The model inference runs on the driver inside foreachBatch (consistent with PyTorch usage).
  - Writes inference results (one JSON per window) to OUT_TOPIC.
"""

import os
import sys
import json
import traceback
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# --- project imports (ensure /app is in PYTHONPATH inside container) ---
sys.path.append("/app")

from src.utils.common import load_json, sliding_windows, window_right_edge_timestamps
from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger
from src.config_entities.config_entity import ModelTrainingConfig
from src.config_entities.artifact_entity import ModelTrainingArtifact, DataProcessingArtifact

# Import your already-defined AEEnsemblePredictor class (exact same code as in batch)
from src.model_components.inference_model import AEEnsemblePredictor  # adjust path if necessary

# -------------------------------------------------------------------
# Config via env
# -------------------------------------------------------------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BROKER", "kafka:9092")
RAW_TOPIC = os.getenv("RAW_TOPIC", "raw_events")
OUT_TOPIC = os.getenv("OUT_TOPIC", "anomaly_predictions")

DATA_PROC_ARTIFACT_PATH = os.getenv("DATA_PROC_ARTIFACT_PATH", "/app/artifacts/data_processing.json")
MODEL_TRAIN_ARTIFACT_PATH = os.getenv("MODEL_TRAIN_ARTIFACT_PATH", "/app/artifacts/model_training.json")
WINDOW_CONFIG_PATH = os.getenv("WINDOW_CONFIG_PATH", "/app/artifacts/window_config.json")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/artifacts/preprocessor.pkl")

TRIGGER_INTERVAL_SEC = int(os.getenv("TRIGGER_INTERVAL_SEC", "5"))
CHECKPOINT_LOCATION = os.getenv("SPARK_CHECKPOINT", "/tmp/spark_checkpoints/anomaly_stream")

# -------------------------------------------------------------------
# Helper loaders (thin wrappers to use same artifacts as batch)
# -------------------------------------------------------------------
def load_window_cfg(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Window config not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

def load_scaler(path: str):
    if not os.path.exists(path):
        return None
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

# -------------------------------------------------------------------
# Spark streaming class
# -------------------------------------------------------------------
class SparkAnomalyStreamer:
    def __init__(
        self,
        spark_app_name: str = "AnomalyStreamingInference",
        data_processing_artifact: DataProcessingArtifact,
        model_training_artifact: ModelTrainingArtifact,
        model_training_config: ModelTrainingConfig
        model_training_dir: str = "/app/artifacts/models",
        device: str = "cpu",
    ) -> None:
        # load configs/artifacts exactly like batch
        try:
            self.model_training_config = model_training_config
            self.model_training_artifact = model_training_artifact
            self.data_processing_artifact = data_processing_artifact
            self.device = model_training_config.device


            logger.info("Initializing SparkAnomalyStreamer...")
            window_cfg_path = self.data_processing_artifact.window_config_path
            scaler_path = os.path.join(self.data_processing_artifact.preprocessor)
            model_config_path = os.path.join(self.model_training_artifact.model_config_path)
            ensemble_meta_path = os.path.join(self.model_training_artifact.ensemble_meta_path)
            

            # compose artifact objects just like batch pipeline expects
            self.data_processing_artifact = DataProcessingArtifact(
                preprocessor=scaler_path,
                window_config_path=window_cfg_path
            )
            self.model_training_artifact = ModelTrainingArtifact(
                model_config_path=model_config_path,
                ensemble_meta_path=ensemble_meta_path
            )
            self.model_training_config = ModelTrainingConfig(
                model_training_dir=model_training_dir,
                device=device
            )

            # instantiate AEEnsemblePredictor exactly like in batch
            logger.info("Loading AEEnsemblePredictor (models + meta)...")
            self.predictor = AEEnsemblePredictor(
                data_processing_artifact=self.data_processing_artifact,
                model_training_artifact=self.model_training_artifact,
                model_training_config=self.model_training_config
            )
            logger.info("AEEnsemblePredictor loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize SparkAnomalyStreamer: {e}")
            logger.error(traceback.format_exc())
            raise

        # Spark session will be created later in start()
        self.spark = None

    # ----------------------
    # microbatch processor (runs on driver)
    # ----------------------
    def _process_microbatch(self, batch_df: DataFrame, epoch_id: int) -> None:
        """
        Called by foreachBatch. Runs on driver, so safe to use PyTorch models here.
        Convert micro-batch -> pandas -> group by source (if present) -> call batch predictor.
        Emit results to Kafka OUT_TOPIC.
        """
        if batch_df.rdd.isEmpty():
            logger.debug(f"Epoch {epoch_id}: empty micro-batch")
            return

        try:
            # convert Kafka value->string
            pdf = batch_df.selectExpr("CAST(value AS STRING) as value_str").toPandas()
            if pdf.empty:
                logger.debug(f"Epoch {epoch_id}: no rows after cast")
                return

            # parse JSON messages
            parsed = pdf["value_str"].apply(json.loads).tolist()
            df = pd.json_normalize(parsed)

            # ensure timestamp col exists
            feature_order = list(self.window_cfg["feature_order"])
            time_col = self.window_cfg.get("timestamp_col", "time")
            if time_col not in df.columns:
                # create synthetic monotonic timestamps if missing
                df[time_col] = pd.date_range("1970-01-01", periods=len(df), freq="S").astype(str)

            # if producer nested features under 'features', expand them
            if "features" in df.columns:
                feats = pd.json_normalize(df["features"])
                df = pd.concat([df.drop(columns=["features"]), feats], axis=1)

            # check presence of required features
            missing = [c for c in feature_order if c not in df.columns]
            if missing:
                logger.warning(f"Epoch {epoch_id}: missing feature columns {missing}. Skipping microbatch.")
                return

            # Group by 'source' column if provided, otherwise process entire df as single sequence
            group_col = "source" if "source" in df.columns else None
            out_messages = []  # collect JSON strings to publish

            def _process_group(gdf: pd.DataFrame, group_key=None):
                # predictor expects a full pandas DataFrame with time + feature columns
                try:
                    # Ensure ordering by time
                    try:
                        gdf = gdf.sort_values(by=time_col)
                    except Exception:
                        pass

                    # call predictor exactly as batch
                    # predictor.predict_from_dataframe will create windows, compute ensemble scores, flags, and return dict
                    result = self.predictor.predict_from_dataframe(gdf)
                    preds = result.get("predictions", [])
                    # attach source if group present
                    for p in preds:
                        if group_key is not None:
                            p["source"] = group_key
                        out_messages.append(json.dumps(p, default=str))
                except Exception as e:
                    logger.error(f"Group {group_key} inference error: {e}")
                    logger.error(traceback.format_exc())

            if group_col:
                for key, g in df.groupby(group_col):
                    _process_group(g, group_key=key)
            else:
                _process_group(df, group_key=None)

            if len(out_messages) == 0:
                logger.info(f"Epoch {epoch_id}: no prediction messages generated.")
                return

            # Build spark DataFrame with value column of JSON strings and write to Kafka
            out_pdf = pd.DataFrame({"value": out_messages})
            out_sdf = self.spark.createDataFrame(out_pdf)
            (out_sdf
                .selectExpr("CAST(value AS STRING) AS value")
                .write
                .format("kafka")
                .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
                .option("topic", OUT_TOPIC)
                .save()
            )
            logger.info(f"Epoch {epoch_id}: wrote {len(out_messages)} messages to {OUT_TOPIC}")

        except Exception as e:
            logger.error(f"Error in _process_microbatch epoch {epoch_id}: {e}")
            logger.error(traceback.format_exc())

    # ----------------------
    # Start streaming
    # ----------------------
    def start(self, master: str = "local[*]") -> None:
        # create spark session
        self.spark = SparkSession.builder \
            .appName("SparkAnomalyStreamer") \
            .config("spark.sql.shuffle.partitions", "2") \
            .master(master) \
            .getOrCreate()

        sc = self.spark.sparkContext
        # we keep scaler as python object on driver; not broadcasted since predictor uses internal scaler
        logger.info("Starting streaming read from Kafka...")
        df = (self.spark.readStream.format("kafka")
              .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
              .option("subscribe", RAW_TOPIC)
              .option("startingOffsets", "latest")
              .load())

        # Use foreachBatch with bound method (works on driver)
        query = (df.writeStream
                 .foreachBatch(lambda batch_df, epoch_id: self._process_microbatch(batch_df, epoch_id))
                 .option("checkpointLocation", CHECKPOINT_LOCATION)
                 .trigger(processingTime=f"{TRIGGER_INTERVAL_SEC} seconds")
                 .start())

        logger.info("SparkAnomalyStreamer started. Awaiting micro-batches...")
        query.awaitTermination()


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
def main():
    try:
        streamer = SparkAnomalyStreamer()
        streamer.start()
    except Exception as e:
        logger.error(f"Fatal error starting SparkAnomalyStreamer: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
