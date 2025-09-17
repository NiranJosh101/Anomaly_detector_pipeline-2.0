import os
import sys
import pandas as pd

from src.config_entities.config_entity import ModelTrainingConfig, DataProcessingConfig, TrainPipelineConfig
from src.config_entities.artifact_entity import ModelTrainingArtifact, DataProcessingArtifact, ModelInferenceArtifact

from src.model_components.LSTM_ARCHITECTURE.lstm_AE import LSTMAutoencoder
from src.model_components.GRU_ARCHITECTURE.gru_AE import GRUAutoencoder
from src.model_components.CNN1D_ARCHITECTURE.cnn_1d_AE import CNNAutoencoder1D

from src.components.prediction_compt.model_inference import AEEnsemblePredictor
from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger

class ModelInference:
    def __init__(self):
        pass
    
    def get_detector(self) -> AEEnsemblePredictor:
        try:
            # Load configs & artifacts
            data_processing_config = DataProcessingConfig(training_pipeline_config=TrainPipelineConfig())
            data_processing_artifact = DataProcessingArtifact(
                train_dataset_path=data_processing_config.train_dataset_path,
                val_dataset_path=data_processing_config.val_dataset_path,
                test_dataset_path=data_processing_config.test_dataset_path,
                window_config_path=data_processing_config.window_config_data_path,
                timestamp_col_path=data_processing_config.timestamp_train_data_path,
                preprocessor=data_processing_config.data_processor_obj_path,
            )

            model_train_config = ModelTrainingConfig(training_pipeline_config=TrainPipelineConfig())
            model_training_artifact = ModelTrainingArtifact(
                lstm_weights_path=model_train_config.lstm_model_path,
                gru_weights_path=model_train_config.gru_model_path,
                cnn_weights_path=model_train_config.cnn_model_path,
                ensemble_meta_path=model_train_config.ensemble_meta_path,
                scaler_pkl_path=data_processing_artifact.preprocessor,
                model_config_path=model_train_config.model_config_path,
            )

            model_training_config = ModelTrainingConfig(training_pipeline_config=TrainPipelineConfig())

            # Build predictor
            detector = AEEnsemblePredictor(
                data_processing_artifact=data_processing_artifact,
                model_training_artifact=model_training_artifact,
                model_training_config=model_training_config
            )

            logger.logging.info("<<====== Initiate model inference =======>>")
            return detector

        except Exception as e:
            raise AnomalyDetectionException(e, sys)

