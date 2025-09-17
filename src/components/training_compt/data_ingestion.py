import os
import sys
import shutil
import pandas as pd
from datetime import datetime
from src.exception_setup.exception import AnomalyDetectionException

from src.config_entities.config_entity import DataIngestionConfig
from src.config_entities.artifact_entity import DataIngestionArtifact
from src.logging_setup import logger

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        
    def validate_uploaded_data(self) -> bool:
        """
        Validates if uploaded CSV is time series data.
        Conditions:
        1. Must have a datetime column (≥ 80% valid datetime values in range)
        2. Must have at least one numeric column
        If validation fails, moves file to 'rejected_data' folder.
        """
        try:
            logger.logging.info("Starting data validation for uploaded file...")

            uploaded_file_path = os.path.join(
                self.data_ingestion_config.data_ingestion_dir,
                self.data_ingestion_config.uploaded_dir,
                self.data_ingestion_config.uploaded_data_name
            )

            if not os.path.exists(uploaded_file_path):
                raise FileNotFoundError(f"Uploaded file not found at {uploaded_file_path}")

            df = pd.read_csv(uploaded_file_path)

            datetime_columns = []
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check plausible UNIX timestamps or year range
                    min_val, max_val = df[col].min(), df[col].max()
                    if not (
                        (1_000_000_000 <= min_val <= 2_000_000_000 and 1_000_000_000 <= max_val <= 2_000_000_000) or
                        (1900 <= min_val <= 2100 and 1900 <= max_val <= 2100)
                    ):
                        continue

                parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() > 0.8:
                    # Extra safeguard — avoid weird ancient or far-future dates
                    if parsed.min().year > 1900 and parsed.max().year < 2100:
                        datetime_columns.append(col)

            # If fails validation → move to rejected folder
            if not datetime_columns or not numeric_cols:
                rejected_dir = os.path.join(
                    self.data_ingestion_config.data_ingestion_dir,
                    self.data_ingestion_config.failed_dir
                )
                os.makedirs(rejected_dir, exist_ok=True)

                rejected_file_path = os.path.join(
                    rejected_dir,
                    self.data_ingestion_config.invalid_csv_name
                )
                shutil.move(uploaded_file_path, rejected_file_path)

                logger.logging.warning(
                    f"Dataset failed validation. "
                    f"Found datetime cols: {datetime_columns}, numeric cols: {numeric_cols}. "
                    f"Moved to rejected_data: {rejected_file_path}"
                )

                # Stop the pipeline completely
                raise ValueError(
                    f"Pipeline stopped: Uploaded dataset is invalid for time series. "
                    f"Moved to: {rejected_file_path}"
                )

           
            return True

        except Exception as e:
            raise AnomalyDetectionException(e, sys)

                
    def export_data_into_validated_store(self) -> str:
        """
        Copies validated data into processed/completed/validated_dataset.csv
        """
        try:
            validated_dir = os.path.join(
                self.data_ingestion_config.processed_dir,
                self.data_ingestion_config.completed_dir,
                self.data_ingestion_config.valid_dataset_name,

            )
            os.makedirs(validated_dir, exist_ok=True)

            src_path = os.path.join(
                self.data_ingestion_config.data_ingestion_dir,
                self.data_ingestion_config.uploaded_dir,
                self.data_ingestion_config.uploaded_data_name
            )

            dest_path = os.path.join(validated_dir, self.data_ingestion_config.valid_dataset_name)
            shutil.copy(src_path, dest_path)

            logger.logging.info(f"Validated data copied to {dest_path}")
            return dest_path

        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates data ingestion:
        1. Validate uploaded file
        2. Export to validated store if valid
        3. Return artifact
        """
        try:
            logger.logging.info("Initiating Data Ingestion Pipeline...")

            if not self.validate_uploaded_data():
                return DataIngestionArtifact(
                    raw_data_file_path=None,
                )

            validated_path = self.export_data_into_validated_store()

            dataingestionartifact = DataIngestionArtifact(
                raw_data_file_path=validated_path
            )

            logger.logging.info(f"Data Ingestion Artifact: {dataingestionartifact}")
            return dataingestionartifact

        except Exception as e:
            raise AnomalyDetectionException(e, sys)
