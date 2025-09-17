import sys

from src.components.training_compt.data_ingestion import DataIngestion
from src.components.training_compt.data_processing import DataProcessing
from src.config_entities.config_entity import DataIngestionConfig, DataProcessingConfig
from src.config_entities.artifact_entity import DataIngestionArtifact, DataProcessingArtifact
from src.config_entities.config_entity import TrainPipelineConfig


from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger



if __name__ == "__main__":
    try:
        trainingpipeline_config = TrainPipelineConfig()

        # Data Ingestion
        data_ingestion_config = DataIngestionConfig(trainingpipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logger.logging.info("<<======Initiate the data ingestion=======>>")
        dataingestionartifact= data_ingestion.initiate_data_ingestion()
        logger.logging.info("<<======Data Ingestion Complete=======>>")


       
    except Exception as e:
            raise AnomalyDetectionException(e, sys)


    
