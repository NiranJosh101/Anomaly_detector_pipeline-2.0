import os
import sys
import numpy as np
import pandas as pd
import torch


PIPELINE_NAME = "ANnomalyDetectionPipeline"
ARTIFACT_DIR_NAME = "Artifact"
DATA_DIR_NAME = "data"


DATA_UPLOADED_NAME = "UploadedData.csv"
DATA_PROCESSED_NAME = "ProcessedData.csv"
DATA_PROCESS_COMPLETED_NAME = "ValidatedData.csv"
DATA_PROCESS_FAILED_NAME = "InvalidData.csv"
DATA_UPLOADED_DIR_NAME = "Uploaded"
DATA_PROCESSED_DIR_NAME = "Processed"
DATA_PROCESS_COMPLETED_DIR_NAME = "valid_dataset"
DATA_PROCESS_FAILED_DIR_NAME = "invalid_dataset"


DATA_PROCESSING_DIR_NAME = "DataProcessing"
DATA_PREPROCESSING_PIPELINE_OBJ_DIR = "preprocessing_obj"
DATA_PREPROCESSING_PIPELINE_OBJ_NAME = "preprocessing_pipeline.pkl"
DATA_PROCESSING_DATA_TRANSFORM_DIR_NAME = "data_transform"
DATA_PROCESSING_X_TRAIN_TRANSFORM_NAME = "X_train.npy"
DATA_PROCESSING_X_VAL_TRANSFORM_NAME = "X_val.npy"
DATA_PROCESSING_X_TEST_TRANSFORM_NAME = "X_test.npy"
DATA_PROCESSING_TS_TRANSFORM_NAME = "ts_train.npy"
DATA_PROCESSING_TS_VAL_TRANSFORM_NAME = "ts_val.npy"
DATA_PROCESSING_TS_TEST_TRANSFORM_NAME = "ts_test.npy"
DATA_PROCESSING_WINDOW_CONFIG_DIR_NAME = "window_config"
DATA_PROCESSING_WINDOW_CONFIG_NAME = "window_config.json"
DATA_PROCESSING_DATALOADER_DIR_NAME = "dataset"
DATA_PROCESSING_TRAIN_DATASET_NAME = "train_dataset.pkl"
DATA_PROCESSING_VAL_DATASET_NAME = "val_dataset.pkl"
DATA_PROCESSING_TEST_DATASET_NAME = "test_dataset.pkl"
DATA_PROCESSING_TIMESTAMP_DIR_NAME = "timestamp_data"
DATA_PROCESSING_TRAIN_TIMESTAMP_NAME = "train_timestamp.pkl"
DATA_PROCESSING_VAL_TIMESTAMP_NAME = "val_timestamp.pkl"
DATA_PROCESSING_TEST_TIMESTAMP_NAME = "test_timestamp.pkl"
WINDOW_SIZE = 16            
WINDOW_STEP = 1  
VAL_SPLIT_RATIO = 0.2           
TEST_SPLIT_RATIO = 0.2
BATCH_SIZE = 20
SHUFFLE_TRAIN = True
NUM_WORKERS = 2
PIN_MEMORY = True


MODEL_FINAL_MODEL_DIR_NAME = "final_models"
MODEL_DATALOADER_DIR_NAME = "model_dataloader"
MODEL_TRAINING_TRAIN_DATALOADER = "train_dataloader.pkl"
MODEL_TRAINING_VAL_DATALOADER = "val_dataloader.pkl"
MODEL_TRAINING_TEST_DATALOADER = "test_dataloader.pkl"
MODEL_TRAINING_DIR_NAME = "model_training"
MODEL_LSTM_WEIGHTS_NAME = "lstm_autoencoder.pt"
MODEL_GRU_WEIGHTS_NAME = "gru_autoencoder.pt"
MODEL_CNN_WEIGHTS_NAME = "gru_autoencoder.pt"
MODEL_VAL_ERROR_LSTM = "val_error_lstm.npy"
MODEL_VAL_ERROR_ENSEMBLE = "val_errors_ensemble.npy"
MODEL_VAL_ERROR_GRU = "val_error_gru.npy"
MODEL_VAL_ERROR_CNN = "val_error_cnn.npy"
MODEL_ESEMBLE_META_NAME = "ensemble_meta.json"
MODEL_TEST_ERROR_LSTM = "test_error_lstm.npy"
MODEL_TEST_ERROR_GRU = "test_error_gru.npy" 
MODEL_TEST_ERROR_CNN = "test_error_cnn.npy"
MODEL_TEST_ERRORS_ENSEMBLE = "test_errors_ensemble.npy"
MODEL_TEST_FLAGS_ENSEMBLE = "test_flags_ensemble.npy"
MODEL_ENSEMBLE_TEST_RESULTS = "ensemble_test_results.json"
MODEL_TRAINING_MODEL_CONFIG = "model_config.json"
MODEL_TRAINING_BATCH_SIZE = 32
MODEL_TRAINING_NUM_EPOCH = 10
MODEL_TRAINING_LR = 1e-3
MODEL_WEIGHT_DECAY = 1e-5
MODEL_GRAD_CLIP = 1
MODEL_TRAINING_ENSEMBLE_NORMALIZATION = "zscore"
MODEL_TRAINING_THRESHOLD_METHOD = "quantile"
MODEL_TRAINING_THRESHOLD_Q = 0.99
MODEL_TRAINING_THRESHOLD_K = 3
MODEL_USE_AMP = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




MODEL_INFERENCE_DIR_NAME = "model_inference"
MODEL_INFERENCE_RESULTS_FILE = "inference_results.csv"





