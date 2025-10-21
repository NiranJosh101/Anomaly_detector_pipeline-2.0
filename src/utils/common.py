import json
from typing import Dict, Tuple, Optional, List, Any, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_object_dtype,
)
from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger
import os, sys
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src.constants.trainingpipeline import MODEL_TRAINING_THRESHOLD_METHOD, MODEL_TRAINING_THRESHOLD_K, MODEL_TRAINING_THRESHOLD_Q





def detect_timestamp_column(
    df: pd.DataFrame,
    threshold: float = 0.8,
    candidate_names: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """
    Detect which column most likely contains timestamps.

    Priority:
      1) Explicit candidate_names (if provided)
      2) datetime64 columns
      3) object/string columns that parse to datetimes
      4) numeric columns that look like unix timestamps (guarded)

    Returns column name or None if confidence < threshold.
    """
    if df is None or df.shape[1] == 0:
        return None

    now = pd.Timestamp.now()
    earliest = pd.Timestamp("1900-01-01")               
    latest = now + pd.DateOffset(years=50)

    total = len(df)
    best_col = None
    best_score = 0.0

    def frac_in_range(ts: pd.Series) -> float:
        s = ts.dropna()
        if s.empty:
            return 0.0
        mask = (s >= earliest) & (s <= latest)
        return float(mask.sum()) / float(total)

    def parse_object(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

    def is_mono(ts: pd.Series) -> bool:
        s = pd.Series(ts).dropna()
        return bool(s.is_monotonic_increasing or s.is_monotonic)

    if candidate_names:
        for name in candidate_names:
            if name in df.columns:
                col = df[name]
                if is_datetime64_any_dtype(col):
                    return name
                if col.dtype.kind in ("O", "U", "S"):
                    parsed = parse_object(col)
                    if (parsed.notna().mean() >= threshold):
                        return name
                    
                if np.issubdtype(col.dtype, np.number):
                    series = col.dropna()
                    if not series.empty:
    
                        if series.median() >= 10_000_000:  
                            s_sec = pd.to_datetime(series, unit="s", errors="coerce")
                            if (s_sec.notna().mean() >= threshold):
                                return name
                        if series.median() >= 1_000_000_000_000:  
                            s_ms = pd.to_datetime(series, unit="ms", errors="coerce")
                            if (s_ms.notna().mean() >= threshold):
                                return name

    
    dt_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
    if dt_cols:
        
        best = None
        best_dt_score = -1.0
        for c in dt_cols:
            col = df[c]
            score = frac_in_range(col)
            if is_mono(col):
                score += 0.2  #
            if score > best_dt_score:
                best_dt_score = score
                best = c
        return best

    #
    for c in df.select_dtypes(include=["object", "string"]).columns:
        try:
            parsed = parse_object(df[c])
            score = parsed.notna().mean()
            
            score += 0.2 * frac_in_range(parsed)
            if is_mono(parsed):
                score += 0.1
        except Exception:
            score = 0.0

        if score > best_score:
            best_score = score
            best_col = c

    if best_col and best_score >= threshold:
        return best_col

    
    for c in df.select_dtypes(include=["number"]).columns:
        series = df[c].dropna()
        if series.empty:
            continue

       
        med = float(series.median())

        score = 0.0
       
        if med >= 10_000_000:  
            s_sec = pd.to_datetime(series, unit="s", errors="coerce")
            score = max(score, s_sec.notna().mean() + 0.2 * frac_in_range(s_sec) + (0.1 if is_mono(s_sec) else 0.0))

        
        if med >= 1_000_000_000_000:  
            s_ms = pd.to_datetime(series, unit="ms", errors="coerce")
            score = max(score, s_ms.notna().mean() + 0.2 * frac_in_range(s_ms) + (0.1 if is_mono(s_ms) else 0.0))

        if score > best_score:
            best_score = score
            best_col = c

    if best_col and best_score >= threshold:
        return best_col

    return None




def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise AnomalyDetectionException(e, sys) from e
    


def save_object(file_path: str, obj: object) -> None:
    try:
        logger.logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise AnomalyDetectionException(e, sys) from e
    


def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise AnomalyDetectionException(e, sys) from e
    


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise AnomalyDetectionException(e, sys) from e
    




def save_json(obj: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)
    except Exception as e:
        raise AnomalyDetectionException(e, sys) from e





def per_window_score(recons: np.ndarray, trues: np.ndarray) -> np.ndarray:
    """
    Compute **per-window** reconstruction error:
        mean of squared error across time and features.
    Shapes:
        recons, trues: (N, L, C)
    Returns:
        scores: (N,)
    """
    return ((recons - trues) ** 2).mean(axis=(1, 2))



def per_timestep_errors(recons: np.ndarray, trues: np.ndarray) -> np.ndarray:
    """
    Compute **per-timestep** error (MSE over features per time step):
        (N, L)
    Useful later for explainability / localization.
    """
    return ((recons - trues) ** 2).mean(axis=2)



def pick_ensemble_threshold(
    scores: np.ndarray,
    method: str = MODEL_TRAINING_THRESHOLD_METHOD,
    q: float = MODEL_TRAINING_THRESHOLD_Q,
    k: float = MODEL_TRAINING_THRESHOLD_Q
) -> float:
    """
    Learn a **single threshold** from ensemble scores (VALIDATION set).
      method="quantile" -> threshold = quantile(scores, q)
      method="std"      -> threshold = mean(scores) + k * std(scores)
    """
    if method == "quantile":
        return float(np.quantile(scores, q))
    elif method == "std":
        return float(scores.mean() + k * scores.std())
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    

def _tensor_to_numpy(x):
    """Robustly convert torch or numpy or list to numpy array (dtype=object for mixed)."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        try:
            return np.array(x)
        except Exception:
            return np.array(x, dtype=object)
    if torch.is_tensor(x):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            return np.array(x, dtype=object)
    # fallback
    return np.array(x, dtype=object)


def load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise AnomalyDetectionException(e, sys) from e

    
    
def mse_per_window(recons: np.ndarray, trues: np.ndarray) -> np.ndarray:
    return ((recons - trues) ** 2).mean(axis=(1, 2))


def sliding_windows(arr_2d: np.ndarray, W: int, step: int) -> np.ndarray:
    T, C = arr_2d.shape
    if T < W:
        return np.empty((0, W, C), dtype=arr_2d.dtype)
    starts = np.arange(0, T - W + 1, step, dtype=int)
    return np.stack([arr_2d[i:i+W] for i in starts], axis=0)



def window_right_edge_timestamps(ts: np.ndarray, W: int, step: int) -> np.ndarray:
    if ts is None:
        return None
    T = len(ts)
    if T < W:
        return np.array([], dtype=object)
    starts = np.arange(0, T - W + 1, step, dtype=int)
    return ts[starts + (W - 1)]



def compute_ensemble_from_model_errors(
    errs_by_model: Dict[str, np.ndarray],
    normalization: str,
    normalizers: Optional[Dict[str, Dict[str, float]]]
) -> np.ndarray:
    names = sorted(errs_by_model.keys())
    mat = np.vstack([errs_by_model[n] for n in names])  # (M, N)

    if normalization == "none":
        return mat.mean(axis=0)

    if normalization == "zscore":
        if not normalizers:
            raise ValueError("Normalization='zscore' requires normalizers in ensemble_meta.json")
        zs = []
        for n in names:
            mu = normalizers[n]["mean"]
            sd = normalizers[n]["std"] if normalizers[n]["std"] > 0 else 1.0
            zs.append((errs_by_model[n] - mu) / sd)
        return np.vstack(zs).mean(axis=0)

    raise ValueError(f"Unknown normalization: {normalization}")



def batched_reconstruction_errors(model, Xw: np.ndarray, device, batch_size: int = 128) -> np.ndarray:
    errs = []
    with torch.no_grad():
        for i in range(0, len(Xw), batch_size):
            xb = torch.from_numpy(Xw[i:i+batch_size]).to(device).float()
            yhat = model(xb)
            batch_errs = mse_per_window(yhat.cpu().numpy(), Xw[i:i+batch_size])
            errs.append(batch_errs)
    return np.concatenate(errs, axis=0)


def _ensure_dir_for(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _cfg_path(cfg: Any, attr: str, default_filename: str) -> str:
    """Get a path from config, falling back to model_training_dir/default_filename."""
    if hasattr(cfg, attr):
        return getattr(cfg, attr)
    base = getattr(cfg, "model_training_dir", "./models")
    return os.path.join(base, default_filename)


def load_window_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Window config not found at {path}")
    with open(path, "r") as f:
        return json.load(f)
        

def load_scaler(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)