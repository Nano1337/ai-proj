import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# Load your dataset
df = pd.read_csv('data.txt', sep='\t', encoding='utf-16')
# df = df.sample(frac=1).reset_index(drop=True)
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

k = 5
fold_indices = kfold_indices(X, k)
model = LogisticRegression()
scores = []

# Iterate through each fold for cross-validation
for train_indices, test_indices in fold_indices:
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    fold_score = accuracy_score(y_test, y_pred)
    scores.append(fold_score)

print("K-Fold Cross-Validation Scores:", scores)
print("Mean Accuracy:", np.mean(scores))
