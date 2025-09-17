from datetime import datetime
import os
from src.constants import trainingpipeline


class TrainPipelineConfig:
    def __init__(self, timestamp: datetime = datetime.now()):
     
        timestamp_str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name: str = trainingpipeline.PIPELINE_NAME
        self.artifact_name: str = trainingpipeline.ARTIFACT_DIR_NAME
        self.artifact_dir: str = os.path.join(self.artifact_name)

        self.model_dir: str = os.path.join("final_model")

        self.data_dir_name: str = trainingpipeline.DATA_DIR_NAME
        self.data_dir: str = os.path.join(self.data_dir_name)

        self.timestamp: str = timestamp_str


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainPipelineConfig):

        # Base directory for data ingestion
        self.data_ingestion_dir = os.path.abspath(training_pipeline_config.data_dir)
        self.timestamp = training_pipeline_config.timestamp
        self.uploaded_dir = os.path.join(self.data_ingestion_dir, trainingpipeline.DATA_UPLOADED_DIR_NAME)
        self.processed_dir = os.path.join(self.data_ingestion_dir, trainingpipeline.DATA_PROCESSED_DIR_NAME)
        self.completed_dir = os.path.join(self.processed_dir, trainingpipeline.DATA_PROCESS_COMPLETED_DIR_NAME)
        self.failed_dir = os.path.join(self.processed_dir, trainingpipeline.DATA_PROCESS_FAILED_DIR_NAME)

        self.uploaded_data_name = trainingpipeline.DATA_UPLOADED_NAME
        self.processed_data_name = trainingpipeline.DATA_PROCESSED_NAME
        self.valid_dataset_name = trainingpipeline.DATA_PROCESS_COMPLETED_NAME
        self.invalid_dataset_name = trainingpipeline.DATA_PROCESS_FAILED_DIR_NAME
        self.invalid_csv_name = trainingpipeline.DATA_PROCESS_FAILED_NAME




class DataProcessingConfig:
    def __init__(self, training_pipeline_config: TrainPipelineConfig):
        self.data_processing_dir = os.path.join(training_pipeline_config.artifact_dir, trainingpipeline.DATA_PROCESSING_DIR_NAME)
        self.data_processor_obj_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PREPROCESSING_PIPELINE_OBJ_DIR)
        self.data_processor_obj_path = os.path.join(self.data_processor_obj_dir, trainingpipeline.DATA_PREPROCESSING_PIPELINE_OBJ_NAME)
        self.data_processing_dataset_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_DATALOADER_DIR_NAME)
        self.train_dataset_path = os.path.join(self.data_processing_dataset_dir, trainingpipeline.DATA_PROCESSING_TRAIN_DATASET_NAME)
        self.val_dataset_path = os.path.join(self.data_processing_dataset_dir, trainingpipeline.DATA_PROCESSING_VAL_DATASET_NAME)
        self.test_dataset_path = os.path.join(self.data_processing_dataset_dir, trainingpipeline.DATA_PROCESSING_TEST_DATASET_NAME)
        self.data_tramsform_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_DATA_TRANSFORM_DIR_NAME)
        self.transformed_X_train_path = os.path.join(self.data_tramsform_dir, trainingpipeline.DATA_PROCESSING_X_TRAIN_TRANSFORM_NAME)
        self.transformed_X_val_path = os.path.join(self.data_tramsform_dir, trainingpipeline.DATA_PROCESSING_X_VAL_TRANSFORM_NAME)
        self.transformed_ts_train_path = os.path.join(self.data_tramsform_dir, trainingpipeline.DATA_PROCESSING_TS_TRANSFORM_NAME)
        self.transformed_X_test_path = os.path.join(self.data_tramsform_dir, trainingpipeline.DATA_PROCESSING_X_TEST_TRANSFORM_NAME)
        self.transformed_ts_test_path = os.path.join(self.data_tramsform_dir, trainingpipeline.DATA_PROCESSING_TS_TEST_TRANSFORM_NAME)
        self.timestamp_data_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_TIMESTAMP_DIR_NAME)
        self.timestamp_train_data_path = os.path.join(self.timestamp_data_dir, trainingpipeline.DATA_PROCESSING_TRAIN_TIMESTAMP_NAME)
        self.timestamp_val_data_path = os.path.join(self.timestamp_data_dir, trainingpipeline.DATA_PROCESSING_VAL_TIMESTAMP_NAME)
        self.timestamp_test_data_path = os.path.join(self.timestamp_data_dir, trainingpipeline.DATA_PROCESSING_TEST_TIMESTAMP_NAME)
        self.window_config_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_WINDOW_CONFIG_DIR_NAME)
        self.window_config_data_path = os.path.join(self.window_config_dir, trainingpipeline.DATA_PROCESSING_WINDOW_CONFIG_NAME)
        self.window_size = trainingpipeline.WINDOW_SIZE
        self.window_step = trainingpipeline.WINDOW_STEP
        self.test_split_ratio = trainingpipeline.TEST_SPLIT_RATIO
        self.val_split_ratio = trainingpipeline.VAL_SPLIT_RATIO
        self.batch_size = trainingpipeline.BATCH_SIZE
        self.shuffle_train = trainingpipeline.SHUFFLE_TRAIN
        self.num_workers = trainingpipeline.NUM_WORKERS
        self.pin_memory = trainingpipeline.PIN_MEMORY



class ModelTrainingConfig:
    def __init__(self, training_pipeline_config: TrainPipelineConfig):
        self.model_training_dir = os.path.join(training_pipeline_config.artifact_dir, trainingpipeline.MODEL_TRAINING_DIR_NAME, trainingpipeline.MODEL_FINAL_MODEL_DIR_NAME)
        self.data_loader_dir = os.path.join(training_pipeline_config.artifact_dir, trainingpipeline.MODEL_DATALOADER_DIR_NAME)
        self.train_dataloader_path = os.path.join(self.data_loader_dir, trainingpipeline.MODEL_TRAINING_TRAIN_DATALOADER)
        self.val_dataloader_path = os.path.join(self.data_loader_dir, trainingpipeline.MODEL_TRAINING_VAL_DATALOADER)
        self.test_dataloader_path = os.path.join(self.data_loader_dir, trainingpipeline.MODEL_TRAINING_TEST_DATALOADER)
        self.lstm_model_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_LSTM_WEIGHTS_NAME)
        self.gru_model_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_GRU_WEIGHTS_NAME)
        self.cnn_model_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_CNN_WEIGHTS_NAME)
        self.val_error_ensemble = os.path.join(self.model_training_dir, trainingpipeline.MODEL_VAL_ERROR_ENSEMBLE)
        self.val_error_lstm_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_VAL_ERROR_LSTM)
        self.val_error_gru_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_VAL_ERROR_GRU)
        self.val_error_cnn_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_VAL_ERROR_CNN)
        self.test_error_ensemble_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_TEST_ERRORS_ENSEMBLE)
        self.test_flags_ensemnle_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_TEST_FLAGS_ENSEMBLE)
        self.esemble_test_results_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_ENSEMBLE_TEST_RESULTS)
        self.test_error_lstm_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_TEST_ERROR_LSTM)
        self.test_error_gru_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_TEST_ERROR_GRU)
        self.test_error_cnn_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_TEST_ERROR_CNN)
        self.ensemble_meta_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_ESEMBLE_META_NAME)
        self.model_config_path = os.path.join(self.model_training_dir, trainingpipeline.MODEL_TRAINING_MODEL_CONFIG)
        self.num_epoch = trainingpipeline.MODEL_TRAINING_NUM_EPOCH
        self.learning_rate = trainingpipeline.MODEL_TRAINING_LR
        self.weight_decay = trainingpipeline.MODEL_WEIGHT_DECAY
        self.grad_clip = trainingpipeline.MODEL_GRAD_CLIP
        self.ensemble_normalization = trainingpipeline.MODEL_TRAINING_ENSEMBLE_NORMALIZATION
        self.threshold_method = trainingpipeline.MODEL_TRAINING_THRESHOLD_METHOD
        self.threshold_q = trainingpipeline.MODEL_TRAINING_THRESHOLD_Q
        self.threshold_k = trainingpipeline.MODEL_TRAINING_THRESHOLD_K
        self.batch_size = trainingpipeline.MODEL_TRAINING_BATCH_SIZE
        self.device = trainingpipeline.DEVICE
        self.use_amp = trainingpipeline.MODEL_USE_AMP

        self.model_inference_dir = os.path.join(training_pipeline_config.artifact_dir, trainingpipeline.MODEL_INFERENCE_DIR_NAME)
        self.inference_results_file = os.path.join(self.model_inference_dir, trainingpipeline.MODEL_INFERENCE_RESULTS_FILE)


