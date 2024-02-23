import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression # TODO: PLACEHOLDER


# Load your dataset
df = pd.read_csv('data.txt', sep='\t', encoding='utf-16')
df = df.sample(frac=1).reset_index(drop=True)
X = df.drop('utility', axis=1).values
y = df['utility'].apply(lambda x: {'very low': 0, 'low': 1, 'average': 2, 'good': 3, 'great': 4}[x]).values

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

# def calculate_mre(y_true, y_pred):
#     mre = np.mean(np.abs((y_true - y_pred) / y_true))
#     return mre

def calculate_mre(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_indices = y_true != 0
    if len(nonzero_indices) > 0:
        return np.mean(np.abs((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices]))
    else:
        return np.nan

k = 5
fold_indices = kfold_indices(X, k)
model = LogisticRegression(max_iter = 1000) # TODO: PLACEHOLDER
scores = []
mres = []

# Iterate through each fold for cross-validation
for train_indices, test_indices in fold_indices:
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    X_train_normalized, X_test_normalized = normalize(X_train, X_test)
    
    model.fit(X_train_normalized, y_train)
    y_pred = model.predict(X_test_normalized) # PLACEHOLDER
    
    fold_score = np.sum(y_test == y_pred) / len(y_test)
    scores.append(fold_score)

    fold_mre = calculate_mre(y_test, y_pred)
    mres.append(fold_mre)

print("K-Fold Cross-Validation Scores:", scores)
print("Mean Accuracy:", np.mean(scores))
print("Mean MRE:", np.nanmean(mres))
print("MRE Variance:", np.nanvar(mres))
