import torch
import numpy as np
import random
import itertools
from scipy.signal import savgol_filter


def set_seed(seed=42):
    # Python base
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    # If using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # These two ensure that the GPU algorithms aren't non-deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_dataset(input_dim=6):
    # Generate every possible combination of 0s and 1s
    table = list(itertools.product([0, 1], repeat=input_dim))
    X = torch.tensor(table).float()

    # Assign a fixed random label to each unique input
    # This is "memorization" - the hardest thing for a model to learn
    y = torch.randint(0, 2, (len(X),)).long()

    return X, y


def uniform_derivative(y, x, window_size=5, poly_order=1):
    """
    Robust derivative
    window_size: Must be odd. Larger = smoother but may blunt peaks.
    poly_order: Usually 2 or 3.
    """
    # deriv=1 tells the filter to compute the first derivative
    # delta is the spacing between x points (assumes uniform spacing)
    dx = x[1] - x[0]
    return savgol_filter(y, window_size, poly_order, deriv=1, delta=dx)


def general_derivative(y, x):
    """Method for calculating the derivative in the general case."""
    out = []
    for i in range(len(x)):
        j1 = max(i-1, 0)
        j2 = min(i+1, len(x)-1)
        dx = x[j2] - x[j1]
        dy = y[j2] - y[j1]
        if dx == 0:
            out.append(0)
        else:
            out.append(dy / dx)
    return np.array(out)


def clip_to_nan(arr, a_min, a_max):
    return np.where((arr >= a_min) & (arr <= a_max), arr, np.nan)
