from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    raw_data_file_path: str



@dataclass
class DataProcessingArtifact:
    train_dataset_path: str
    val_dataset_path: str
    test_dataset_path: str
    window_config_path: str
    timestamp_col_path: str
    preprocessor: str


@dataclass
class ModelTrainingArtifact:
    lstm_weights_path: str
    gru_weights_path: str
    cnn_weights_path: str
    ensemble_meta_path: str
    scaler_pkl_path: str
    model_config_path: str



@dataclass
class ModelInferenceArtifact:
    results_path: str

    