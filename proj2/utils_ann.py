import pandas as pd
import numpy as np

def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        if i < k - 1:
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
        else:
            test_indices = indices[i * fold_size:]  # last fold included
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds

def normalize(train, test):
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_normalized = (train - mean) / std
    test_normalized = (test - mean) / std
    return train_normalized, test_normalized
