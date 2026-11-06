import warnings
from pathlib import Path
import random, inspect, json
import torch
import numpy as np
from datetime import datetime
import pytz, pickle, gzip, os
from typing import Tuple, List, Optional
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

__all__ = [
    '_round',
    'load_project_root',
    'create_folder',
    'datetime_now',
    'fix_seed',
    'auto_repr',
    'load_json',
    'load_from_pickle',
    'list_to_onehot',
    'list_to_onehot_oov',
    'print_results'
]

def _round(x, decimal):
    if isinstance(x, (str, float, int)):
        return float(f"{x:.{decimal}f}")
    elif isinstance(x, (list, np.ndarray, tuple, torch.Tensor)):
        return [float(f"{_x:.{decimal}f}") for _x in x]

def load_project_root(N=1):
    return Path(__file__).resolve().parents[N]

def create_folder(folder_name: str) -> str:
    
    if len(folder_name):
        if not os.path.isdir(folder_name):
            print(f'Creating Folder:{folder_name}')
            os.makedirs(folder_name)
        
    return folder_name

def datetime_now(path: bool = True):
    """
    Returns current datetime in Asia/Kolkata timezone.

    Returns:
        dt   : full formatted datetime string
        date : YYYY_MM_DD
        time : HH_MM_SS
    """
    ist = pytz.timezone("Asia/Kolkata")

    now = datetime.now(ist)
    dt = now.strftime("%Y%m%d_%H_%M_%S" if path else "%Y/%m/%d [%H:%M:%S]")
    date = now.strftime("%Y_%m_%d")
    time = now.strftime("%H_%M_%S")

    return dt, date, time

def fix_seed(np_seed=30, torch_seed=30, random_seed=30):
    
    random.seed(random_seed)
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def list_to_onehot(element, elements_list, val=1.0) -> list:
    
    onehot = [0.]*len(elements_list)
    idx = elements_list.index(element)
    onehot[idx] = val
    
    return onehot

def list_to_onehot_oov(element, elements_list, val=1.0):
    """
    One-hot encoder with an OOV bucket.
    - Boolean vector if val is 0 or 1.
    - Float vector otherwise.
    """

    size = len(elements_list)
    use_bool = float(val) in (0.0, 1.0)
    onehot = [False] * (size + 1) if use_bool else [0.0] * (size + 1)

    try:
        idx = elements_list.index(element)
    except:
        idx = size
    onehot[idx] = bool(val) if use_bool else val

    return onehot

def auto_repr(self):
    """
    Generalized __repr__ for classes.
    Constructs a multi-line string showing the class name and __init__ arguments with their current values.
    """
    cls_name = self.__class__.__name__
    try:
        sig = inspect.signature(self.__init__)
        args = [p for p in sig.parameters if p != "self"]
    except Exception:
        args = list(vars(self).keys())

    body = ",\n  ".join(f"{k}={repr(getattr(self, k, None))}" for k in args)
    
    return f"{cls_name}(\n  {body}\n)"


def load_json(path):

    with open(path, 'r') as fp:
        data = json.loads(fp.read())

    return data

def load_from_pickle(filepath: str):
    """
    Load Python object from a pickle file.
    Automatically detects gzip compression (even without .gz extension).
    """
    with open(filepath, "rb") as f:
        magic = f.read(2)
    
    if magic == b"\x1f\x8b":  # gzip magic number
        with gzip.open(filepath, "rb") as fp:
            file_ = pickle.load(fp)
    else:
        with open(filepath, "rb") as fp:
            file_ = pickle.load(fp)
    return file_

def percentage_within_fold_change(y_true, y_pred, fold=2, transform_func=lambda x: x):
    """
    Calculate the percentage of repeated measurements within a given fold range.

    Parameters:
    - y_true (list or numpy array): The true experimental values.
    - y_pred (list or numpy array): The predicted experimental values.
    - fold (float): The fold change range (default is 2).

    Returns:
    - float: Percentage of values within the specified fold range.
    """
    y_true = transform_func(np.array(y_true, dtype=float))
    y_pred = transform_func(np.array(y_pred, dtype=float))
    
    # Ensure there are no zero values to avoid division issues
    y_true[y_true == 0] = 1e-3
    assert np.all(y_true != 0), "True values must not contain zero."
    
    # Calculate fold change
    fold_change = y_pred / y_true 
    max_, min_ = np.round(max(fold_change), 2), np.round(min(fold_change), 2)
    within_fold = (1/fold <= fold_change) & (fold_change <= fold)
    
    # Calculate the percentage
    percentage = round(np.sum(within_fold) / len(y_true)*100, 2)
    
    return percentage, max_, min_

def geometric_mean_fold_error(
    y_true, 
    y_pred, 
    transform_func=lambda x: x,
    use_log_domain=False,
    eps=1e-12
):
    """
    Compute Geometric Mean Fold Error (GMFE).

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    transform_func : callable, optional
        Transformation applied before computation.
    use_log_domain : bool, optional
        If True, assumes inputs are already in log scale (e.g., pKa, logP).
        GMFE = 10 ** MAE.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    gmfe : float
    invalid : int
        Number of excluded samples
    """

    y_true = transform_func(np.asarray(y_true, dtype=float))
    y_pred = transform_func(np.asarray(y_pred, dtype=float))

    # Case 1: Already log-transformed values (preferred in chemistry)
    if use_log_domain:
        mae = np.mean(np.abs(y_pred - y_true))
        return 10 ** mae, 0

    # Case 2: Raw values → use absolute ratio
    valid = (np.abs(y_true) > eps) & (np.abs(y_pred) > eps)

    invalid = np.sum(~valid)
    if invalid > 0:
        warnings.warn(f"{invalid} values ignored due to near-zero entries.")

    y_true_valid = np.abs(y_true[valid])
    y_pred_valid = np.abs(y_pred[valid])

    log_fold_errors = np.abs(np.log10(y_pred_valid / y_true_valid))
    gmfe = 10 ** np.mean(log_fold_errors)

    return gmfe, invalid

def regression_test_metrics(y_true: np.ndarray[float] | List[float],
                 y_pred: np.ndarray[float] | List[float],
                 transform_func=lambda x: x,
                 fold_transform_fn=lambda x:x,
                 use_log_domain_gmfe=False,
                 decimals=2,
                 corr_method='pearson',
            ) -> OrderedDict[str, float]:
    
    '''
    Input:
        - model: Random Forest Model
        - y_true: Array/List Object
        - y_pred: Array/List Object
    
    Output: A dictionary containing MSE, RMSE, MAE and r2 Values.
    '''
    if not isinstance(y_pred, np.ndarray):
        y_pred = transform_func(np.array(y_pred))
    if not isinstance(y_true, np.ndarray):
        y_true = transform_func(np.array(y_true))

    assert y_pred.shape == y_true.shape, f"Shape mismatch: y_pred -> {y_pred.shape}, y_true -> {y_true.shape}"
    
    # Calculate metrics on Test Set
    rmse_test = root_mean_squared_error(y_true, y_pred)
    mse_test = rmse_test**2
    mae_test = mean_absolute_error(y_true, y_pred)
    r2_val_test = r2_score(y_true, y_pred)
    if corr_method == "pearson":
        pcc, _ = pearsonr(y_true, y_pred)
    elif corr_method == "spearman":
        pcc, _ = spearmanr(y_true, y_pred)
    elif corr_method == "kendall":
        pcc, _ = kendalltau(y_true, y_pred)
    else:
        raise ValueError(f"Unknown method '{corr_method}'. Choose from "
                         "['pearson', 'spearman', 'kendall'].")
    
    fold2, _, _ = percentage_within_fold_change(y_true=y_true, y_pred=y_pred, fold=2, transform_func=fold_transform_fn)
    fold3, _, _ = percentage_within_fold_change(y_true=y_true, y_pred=y_pred, fold=3, transform_func=fold_transform_fn)
    fold5, _, _ = percentage_within_fold_change(y_true=y_true, y_pred=y_pred, fold=5, transform_func=fold_transform_fn)
    _gmfe, _ = geometric_mean_fold_error(y_true=y_true, y_pred=y_pred, transform_func=fold_transform_fn, use_log_domain=use_log_domain_gmfe)
    
    
    results = OrderedDict({'mse':round(mse_test, decimals),
                           'r2':round(r2_val_test, decimals),
                           'rmse':round(rmse_test, decimals),
                           'mae':round(mae_test, decimals),
                           'PCC':round(pcc, decimals),
                           'GMFE':round(_gmfe, decimals),
                           'fold2':round(fold2, decimals),
                           'fold3':round(fold3, decimals),
                           'fold5':round(fold5, decimals),
                           })
    
    return results

def print_results(train_results, test_results, val_results):
    
    print('='*70)
    print(f"{'& Data':7} & {'R2':7} & {'RMSE':7} & {'MAE':7} & {'PCC':7} & {'GMFE':7} & {'Fold-2':7} & {'Fold-3':7} & {'Fold-5':7}")
    print('-'*70)
    for k, results in {'& Train': train_results, '& Test': test_results, '& Val': val_results}.items():
        print(f"{k:7} & "
            f"{results['r2']:7.2f} & "
            f"{results['rmse']:7.2f} & "
            f"{results['mae']:7.2f} & "
            f"{results['PCC']:7.2f} & "
            f"{results['GMFE']:7.2f} & "
            f"{results['fold2']:7.2f} & "
            f"{results['fold3']:7.2f} & "
            # f"{results['fold5']:7.2f} {r"\\"}")
            f"{results['fold5']:7.2f} \\\\")
    print('='*70)