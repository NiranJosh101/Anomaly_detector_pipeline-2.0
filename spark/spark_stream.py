import os
import sys
import json
import traceback
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


sys.path.append("/app")

from src.utils.common import load_json, sliding_windows, window_right_edge_timestamps, load_window_cfg, load_scaler
from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger
from src.config_entities.config_entity import ModelTrainingConfig
from src.config_entities.artifact_entity import ModelTrainingArtifact, DataProcessingArtifact


from src.model_components.model_inference import AEEnsemblePredictor  


KAFKA_BOOTSTRAP = os.getenv("KAFKA_BROKER", "kafka:9092")
RAW_TOPIC = os.getenv("RAW_TOPIC", "raw_events")
OUT_TOPIC = os.getenv("OUT_TOPIC", "anomaly_predictions")

TRIGGER_INTERVAL_SEC = int(os.getenv("TRIGGER_INTERVAL_SEC", "5"))
CHECKPOINT_LOCATION = os.getenv("SPARK_CHECKPOINT", "/tmp/spark_checkpoints/anomaly_stream")


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


class SparkAnomalyStreamer:
    def __init__(
        self,
        spark_app_name: str = "AnomalyStreamingInference",
        data_processing_artifact: DataProcessingArtifact,
        model_training_artifact: ModelTrainingArtifact,
        model_training_config: ModelTrainingConfig
        device: str = "cpu",
    ) -> None:
        
        try:
            self.model_training_config = model_training_config
            self.model_training_artifact = model_training_artifact
            self.data_processing_artifact = data_processing_artifact
            self.device = model_training_config.device


            logging.logger.info("Initializing SparkAnomalyStreamer...")
            window_cfg_path = self.data_processing_artifact.window_config_path
            scaler_path = os.path.join(self.data_processing_artifact.preprocessor)
            model_config_path = os.path.join(self.model_training_artifact.model_config_path)
            ensemble_meta_path = os.path.join(self.model_training_artifact.ensemble_meta_path)
            model_training_dir = os.path.join(model_training_dir.final_model_dir)
            
            
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

            logging.logger.info("Loading AEEnsemblePredictor (models + meta)...")
            self.predictor = AEEnsemblePredictor(
                data_processing_artifact=self.data_processing_artifact,
                model_training_artifact=self.model_training_artifact,
                model_training_config=self.model_training_config
            )
            logging.logger.info("AEEnsemblePredictor loaded successfully.")

        except Exception as e:
            logging.logger.info(f"Failed to initialize SparkAnomalyStreamer: {e}")
            logging.logger.info(traceback.format_exc())
            raise

        
        self.spark = None

    
    def _process_microbatch(self, batch_df: DataFrame, epoch_id: int) -> None:
        """
        Called by foreachBatch. Runs on driver, so safe to use PyTorch models here.
        Convert micro-batch -> pandas -> group by source -> call batch predictor.
        Emit results to Kafka OUT_TOPIC.
        """
        if batch_df.rdd.isEmpty():
            logging.logger.info(f"Epoch {epoch_id}: empty micro-batch")
            return

        try:
            
            pdf = batch_df.selectExpr("CAST(value AS STRING) as value_str").toPandas()
            if pdf.empty:
                logging.logger.info(f"Epoch {epoch_id}: no rows after cast")
                return

            
            parsed = pdf["value_str"].apply(json.loads).tolist()
            df = pd.json_normalize(parsed)

            # ensure timestamp col exists
            feature_order = list(self.window_cfg["feature_order"])
            time_col = self.window_cfg.get("timestamp_col", "time")
            if time_col not in df.columns:
                df[time_col] = pd.date_range("1970-01-01", periods=len(df), freq="S").astype(str)

            # expand features column if present
            if "features" in df.columns:
                feats = pd.json_normalize(df["features"])
                df = pd.concat([df.drop(columns=["features"]), feats], axis=1)

            # check presence of required features
            missing = [c for c in feature_order if c not in df.columns]
            if missing:
                logging.logger.warning(f"Epoch {epoch_id}: missing feature columns {missing}. Skipping microbatch.")
                return

            
            group_col = "source" if "source" in df.columns else None
            out_messages = []  

            def _process_group(gdf: pd.DataFrame, group_key=None):
                try:
                    try:
                        gdf = gdf.sort_values(by=time_col)
                    except Exception:
                        pass

                    result = self.predictor.predict_from_dataframe(gdf)
                    preds = result.get("predictions", [])
                   
                    for p in preds:
                        if group_key is not None:
                            p["source"] = group_key
                        out_messages.append(json.dumps(p, default=str))
                except Exception as e:
                    logging.logger.error(f"Group {group_key} inference error: {e}")
                    logging.logger.error(traceback.format_exc())

            if group_col:
                for key, g in df.groupby(group_col):
                    _process_group(g, group_key=key)
            else:
                _process_group(df, group_key=None)

            if len(out_messages) == 0:
                logging.logger.info(f"Epoch {epoch_id}: no prediction messages generated.")
                return

            
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
            logging.logger.info(f"Epoch {epoch_id}: wrote {len(out_messages)} messages to {OUT_TOPIC}")

        except Exception as e:
            logging.logger.error(f"Error in _process_microbatch epoch {epoch_id}: {e}")
            logging.logger.error(traceback.format_exc())

    
    def start(self, master: str = "local[*]") -> None:
        
        self.spark = SparkSession.builder \
            .appName("SparkAnomalyStreamer") \
            .config("spark.sql.shuffle.partitions", "2") \
            .master(master) \
            .getOrCreate()

        sc = self.spark.sparkContext
        logging.logger.info("Starting streaming read from Kafka...")
        df = (self.spark.readStream.format("kafka")
              .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
              .option("subscribe", RAW_TOPIC)
              .option("startingOffsets", "latest")
              .load())

        
        query = (df.writeStream
                 .foreachBatch(lambda batch_df, epoch_id: self._process_microbatch(batch_df, epoch_id))
                 .option("checkpointLocation", CHECKPOINT_LOCATION)
                 .trigger(processingTime=f"{TRIGGER_INTERVAL_SEC} seconds")
                 .start())

        logging.logger.info("SparkAnomalyStreamer started. Awaiting micro-batches...")
        query.awaitTermination()



def main():
    try:
        streamer = SparkAnomalyStreamer()
        streamer.start()
    except Exception as e:
        logging.logger.error(f"Fatal error starting SparkAnomalyStreamer: {e}")
        logging.logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
