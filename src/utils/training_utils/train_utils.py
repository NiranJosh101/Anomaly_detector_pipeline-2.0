import numpy as np
import torch
from torch.utils.data import Dataset
from src.exception_setup.exception import AnomalyDetectionException
import sys



class AnomalyDetectorWindowDataset(Dataset):
    """
    PyTorch Dataset for windowed time-series data (reconstruction models).
    Includes timestamps for later mapping of reconstruction errors.
    """

    def __init__(self, X_path: str, loader_fn):
        """
        Args:
            X_path (str): Path to windowed features (.npy)
            y_path (str): Path to windowed targets (.npy)
            ts_path (str): Path to timestamps (.npy)
            loader_fn (callable): Custom function to load numpy arrays
        """
        try:
            self.X = loader_fn(X_path)            
             
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            X_tensor: torch.FloatTensor
            y_tensor: torch.FloatTensor
            ts_array: numpy array (timestamps for the window)
        """
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        
        return X_tensor
    



def window_data(
    X_arr: np.ndarray,
    window_size: int,
    step_size: int
):
    """
    Create sliding windows for time-series data (for reconstruction models).
    Assumes timestamp column has already been removed.
    """
    X_windows = []
    num_samples = X_arr.shape[0]

    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        X_windows.append(X_arr[start:end])

    return np.array(X_windows)
