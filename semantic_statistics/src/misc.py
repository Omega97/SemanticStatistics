import torch
import numpy as np
import random
import itertools


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