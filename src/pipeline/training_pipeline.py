import os
import sys

from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger

from src.components.training_compt.data_ingestion import DataIngestion
from src.components.training_compt.data_processing import DataProcessing
from src.components.training_compt.model_training import ModelTraining

from src.config_entities.artifact_entity import DataIngestionArtifact, DataProcessingArtifact


from src.config_entities.config_entity import (
    TrainPipelineConfig,
    DataIngestionConfig,
    DataProcessingConfig,
    ModelTrainingConfig,
)



class TrainingPipeline:
    def __init__(self):
        try:
            self.trainingpipeline_config = TrainPipelineConfig()
        except Exception as e:
            raise AnomalyDetectionException(e, sys)

    def start_data_ingestion(self):
        try:
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.trainingpipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            logger.logging.info("<<======Initiate the data ingestion=======>>")
            dataingestionartifact= data_ingestion.initiate_data_ingestion()
            logger.logging.info("<<======Data Ingestion Complete=======>>")
            return dataingestionartifact
        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        

    def start_data_processing(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_processing_config = DataProcessingConfig(training_pipeline_config=self.trainingpipeline_config)
            data_processing = DataProcessing(data_ingestion_artifact, data_processing_config)
            logger.logging.info("<<======Initiate the data processing=======>>")
            data_processing_artifact = data_processing.initiate_data_processing()
            logger.logging.info("<<======Data Processing Complete=======>>")
            return data_processing_artifact
        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        

    def start_model_training(self, data_processing_artifact: DataProcessingArtifact, model_training_config: ModelTrainingConfig):
        try:
            data_processing_artifact = data_processing_artifact
            model_training = ModelTraining(data_processing_artifact, model_training_config)
            logger.logging.info("<<======Initiate the model training=======>>")
            model_training_artifact = model_training.initiate_training_and_build_ensemble()
            logger.logging.info("<<======Model Training Complete=======>>")
            return model_training_artifact  
        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_processing_artifact = self.start_data_processing(data_ingestion_artifact)
            model_training_config = ModelTrainingConfig(training_pipeline_config=self.trainingpipeline_config)
            model_training_artifact = self.start_model_training(data_processing_artifact, model_training_config)
            logger.logging.info("<<======Training Pipeline Complete=======>>")
            # return data_ingestion_artifact
        except Exception as e:
            raise AnomalyDetectionException(e, sys)