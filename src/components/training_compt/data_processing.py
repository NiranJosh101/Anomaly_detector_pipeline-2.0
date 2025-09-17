import sys, os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from torch.utils.data import DataLoader


from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger

from src.config_entities.config_entity import DataProcessingConfig
from src.config_entities.artifact_entity import DataIngestionArtifact, DataProcessingArtifact

from src.utils.common import detect_timestamp_column,save_object,save_numpy_array_data,load_object,load_numpy_array_data,save_json,load_json
from src.utils.training_utils.train_utils import window_data, AnomalyDetectorWindowDataset



class DataProcessing:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_processing_config: DataProcessingConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_processing_config = data_processing_config
        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        

    
    def read_validated_data(self) -> pd.DataFrame:
        """
        Reads the validated data from the ingestion artifact.
        """
        try:
            validated_data_path = self.data_ingestion_artifact.raw_data_file_path
            if not validated_data_path:
                raise ValueError("No validated data path provided.")

            df = pd.read_csv(validated_data_path)
            logger.logging.info(f"Validated data read from {validated_data_path}")
            return df

        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        
        
    def handle_missing_and_invalid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect timestamp column and handle missing/invalid values + duplicates in a time-series-friendly way.
        Returns cleaned DataFrame.
        """
        try:
            
            time_col = detect_timestamp_column(df)
            if time_col is None:
                raise ValueError(
                    "Could not detect timestamp column automatically. "
                    "Please provide a timestamp column or check the data."
                )

            
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col]).copy()

           
            df = df.sort_values(by=time_col).reset_index(drop=True)

            # Remove duplicates 
            before_dup = len(df)
            df = df.drop_duplicates(subset=[time_col], keep="first")
            after_dup = len(df)
            if before_dup != after_dup:
                logger.logging.info(f"Removed {before_dup - after_dup} duplicate rows based on timestamp.")

            # Handle missing values
            if df.isna().sum().sum() > 0:
                
                df = df.ffill().bfill()

                numeric_cols = df.select_dtypes(include=["number"]).columns
                if df[numeric_cols].isna().sum().sum() > 0:
                    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit_direction="both")

    
                df = df.dropna(subset=[time_col]).reset_index(drop=True)

            logger.logging.info(f"Missing, invalid values, and duplicates handled. Time column: {time_col}")
            return df

        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        
    


    def resampling_df_freqencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect timestamp column, check if data needs resampling,
        and resample to inferred frequency if irregular.
        """
        try:
            
            time_col = detect_timestamp_column(df)
            if time_col is None:
                raise ValueError("No timestamp column found.")

           
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

           
            inferred_freq = pd.infer_freq(df[time_col])

            if inferred_freq is None:
                deltas = df[time_col].diff().dropna().value_counts()
                most_common_delta = deltas.index[0]
                inferred_freq = pd.tseries.frequencies.to_offset(most_common_delta).freqstr

            
            expected_range = pd.date_range(
                start=df[time_col].min(),
                end=df[time_col].max(),
                freq=inferred_freq
            )
            is_irregular = len(expected_range) != len(df)

            if is_irregular:
                logger.logging.info(f"Data is irregular. Resampling to frequency '{inferred_freq}'")
                df = df.set_index(time_col).resample(inferred_freq).ffill().reset_index()
            else:
                logger.logging.info("Data is regular. No resampling needed.")

            return df

        except Exception as e:
            raise RuntimeError(f"Error in maybe_resample: {e}")

            

        
    def train_val_test_split_time_series(
            self, df: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform a chronological train/validation/test split for time-series data.
        First split dataset into train+test, then split train into train+val.
        For reconstruction models: y == X.
        Saves timestamp columns separately for later use.
        """
        try:
            time_col = detect_timestamp_column(df)
            if time_col is None:
                raise ValueError("Timestamp column could not be detected.")

            
            df = df.sort_values(by=time_col).reset_index(drop=True)

            n_total = len(df)
            test_ratio = self.data_processing_config.test_split_ratio
            val_ratio = self.data_processing_config.val_split_ratio

            
            n_test = int(n_total * test_ratio)
            n_trainval = n_total - n_test

            trainval_df = df.iloc[:n_trainval].reset_index(drop=True)
            test_df = df.iloc[n_trainval:].reset_index(drop=True)

            
            n_val = int(n_trainval * val_ratio)
            n_train = n_trainval - n_val

            train_df = trainval_df.iloc[:n_train].reset_index(drop=True)
            val_df = trainval_df.iloc[n_train:].reset_index(drop=True)

            logger.logging.info(
                f"Time-series split complete. "
                f"Train size: {len(train_df)}, "
                f"Validation size: {len(val_df)}, "
                f"Test size: {len(test_df)}"
            )

            
            save_object(self.data_processing_config.timestamp_train_data_path, train_df[[time_col]])
            save_object(self.data_processing_config.timestamp_val_data_path, val_df[[time_col]])
            save_object(self.data_processing_config.timestamp_test_data_path, test_df[[time_col]])

          
            X_train_features = train_df.drop(columns=[time_col])
            X_val_features = val_df.drop(columns=[time_col])
            X_test_features = test_df.drop(columns=[time_col])

            return X_train_features, X_val_features, X_test_features

        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

        

    def save_window_config(self, X_train: pd.DataFrame):
        """
        Save window configuration (W, step, feature_order, timestamp_col) 
        as a JSON artifact for reproducibility at inference.
        """
        try:
            
            feature_order = list(X_train.columns)

            train_timestamps = load_object(self.data_processing_config.timestamp_train_data_path)
            timestamp_col = train_timestamps.columns[0]

            window_config = {
                "window_size": self.data_processing_config.window_size,
                "step_size": self.data_processing_config.window_step,
                "feature_order": feature_order,
                "timestamp_col": timestamp_col
            }

            # Save JSON
            save_json(
                obj=window_config,
                file_path=self.data_processing_config.window_config_data_path,
            )

            logger.logging.info(f"Window config saved successfully")

        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e


            
    
    def fit_and_save_preprocessing_pipeline(self, X_train: pd.DataFrame) -> Pipeline:
        """
        Fit the preprocessing pipeline (scaling + encoding only, 
        timestamps excluded), save it for later inference.
        
        """
        try:
            
            numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

           
            
            num_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

        
            preprocessing = ColumnTransformer([
                ("num", num_pipeline, numeric_cols),
                ("cat", cat_pipeline, categorical_cols),
            ], remainder="drop")

            preprocessing.fit(X_train)

            save_object(
                file_path=self.data_processing_config.data_processor_obj_path,
                obj=preprocessing
            )


            logger.logging.info(f"Preprocessing pipeline fitted and saved.")

            return preprocessing

        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e


            
            
    def transform_train_val_test(
            self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
        ):
        """
        Transform train, validation, and test datasets using the saved preprocessing pipeline.
        Assumes timestamp column has already been removed earlier.
        """
        try:
            pipeline_path = self.data_processing_config.data_processor_obj_path
            preprocessor = load_object(pipeline_path)

            # Apply preprocessing
            X_train_arr = preprocessor.transform(X_train)
            X_val_arr = preprocessor.transform(X_val)
            X_test_arr = preprocessor.transform(X_test)

            # Convert back to DataFrames
            X_train_final = pd.DataFrame(X_train_arr)
            X_val_final = pd.DataFrame(X_val_arr)
            X_test_final = pd.DataFrame(X_test_arr)

            logger.logging.info("Train, validation, and test datasets transformed.")

            return X_train_final, X_val_final, X_test_final

        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e
        


    def apply_windowing_to_train_val_test(
            self,
            X_train_df: pd.DataFrame,
            X_val_df: pd.DataFrame,
            X_test_df: pd.DataFrame
        ):
        """
        Apply windowing to train, validation, and test sets for reconstruction models.
        Timestamp column is assumed already removed earlier.
        """
        try:
            # Convert to numpy arrays
            X_train_arr = X_train_df.to_numpy()
            X_val_arr = X_val_df.to_numpy()
            X_test_arr = X_test_df.to_numpy()

            # Apply windowing
            X_train_win = window_data(
                X_train_arr,
                window_size=self.data_processing_config.window_size,
                step_size=self.data_processing_config.window_step
            )
            X_val_win = window_data(
                X_val_arr,
                window_size=self.data_processing_config.window_size,
                step_size=self.data_processing_config.window_step
            )
            X_test_win = window_data(
                X_test_arr,
                window_size=self.data_processing_config.window_size,
                step_size=self.data_processing_config.window_step
            )

            # Save windowed arrays
            save_numpy_array_data(self.data_processing_config.transformed_X_train_path, X_train_win)
            save_numpy_array_data(self.data_processing_config.transformed_X_val_path, X_val_win)
            save_numpy_array_data(self.data_processing_config.transformed_X_test_path, X_test_win)

            logger.logging.info(
                f"Windowing applied successfully. "
                f"Train shape: {X_train_win.shape}, "
                f"Validation shape: {X_val_win.shape}, "
                f"Test shape: {X_test_win.shape}"
            )

            return X_train_win, X_val_win, X_test_win

        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e


        
    
    def create_all_datasets(
            self,
            X_train_path: str = None,
            X_val_path: str = None,
            X_test_path: str = None,
        ):
        """
        Creates timestamp-aware PyTorch Dataset objects for train/val/test.
        Uses provided paths if given, otherwise falls back to config paths.

        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        try:
            # Resolve paths (fall back to config if not provided)
            X_train_path = X_train_path or self.data_processing_config.transformed_X_train_path
            X_val_path = X_val_path or self.data_processing_config.transformed_X_val_path
            X_test_path = X_test_path or self.data_processing_config.transformed_X_test_path

            # Create datasets
            train_dataset = AnomalyDetectorWindowDataset(
                X_path=X_train_path,
                loader_fn=load_numpy_array_data
            )

            val_dataset = AnomalyDetectorWindowDataset(
                X_path=X_val_path,
                loader_fn=load_numpy_array_data
            )

            test_dataset = AnomalyDetectorWindowDataset(
                X_path=X_test_path,
                loader_fn=load_numpy_array_data
            )

            # Save datasets
            save_object(self.data_processing_config.train_dataset_path, train_dataset)
            save_object(self.data_processing_config.val_dataset_path, val_dataset)
            save_object(self.data_processing_config.test_dataset_path, test_dataset)

            logger.logging.info(
                "Train, validation, and test datasets created and saved successfully. "
                f"Train size: {len(train_dataset)}, "
                f"Validation size: {len(val_dataset)}, "
                f"Test size: {len(test_dataset)}"
            )

            return train_dataset, val_dataset, test_dataset

        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e



    def initiate_data_processing(self) -> DataProcessingArtifact:
        """
        Main method to run the entire data processing pipeline.
        """
        try:
            #
            df = self.read_validated_data()
            df = self.handle_missing_and_invalid(df)
            df = self.resampling_df_freqencies(df)

            
            X_train, X_val, X_test = self.train_val_test_split_time_series(df)

            
            self.save_window_config(X_train)

            
            preprocessor = self.fit_and_save_preprocessing_pipeline(X_train)

            
            X_train_arr, X_val_arr, X_test_arr = self.transform_train_val_test(X_train, X_val, X_test)


            X_train_win, X_val_win, X_test_win = self.apply_windowing_to_train_val_test(
                X_train_arr, X_val_arr, X_test_arr
            )

            train_dataset, val_dataset, test_dataset = self.create_all_datasets(
                X_train_path=self.data_processing_config.transformed_X_train_path,
                X_val_path=self.data_processing_config.transformed_X_val_path,
                X_test_path=self.data_processing_config.transformed_X_test_path,
            )

            data_processing_artifact = DataProcessingArtifact(
                train_dataset_path=self.data_processing_config.train_dataset_path,
                val_dataset_path=self.data_processing_config.val_dataset_path,
                test_dataset_path=self.data_processing_config.test_dataset_path,
                window_config_path=self.data_processing_config.window_config_data_path,
                timestamp_col_path=self.data_processing_config.timestamp_train_data_path,
                preprocessor=preprocessor
            )

            logger.logging.info("Data processing pipeline completed successfully.")

            return data_processing_artifact

        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e
