import pandas as pd
import numpy as np


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


k = 5
fold_indices = kfold_indices(X, k)

fold_accuracies = []

# Iterate through each fold for cross-validation
for train_indices, test_indices in fold_indices:
    # reset model and creaet train/val split
    model = model = DecisionTree(max_depth=5) #TODO: NEED TO RUN ON 2, 3, 4, 5
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    X_train, X_test = normalize(X_train, X_test)
    
    # train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) 
    
    # calculate accuracy
    correct_predictions = sum(1 for true, pred in zip(y_test, y_pred_labels) if true == pred)
    accuracy = correct_predictions / len(y_test)
    
    fold_accuracies.append(accuracy)

mean_accuracy = np.mean(fold_accuracies)
accuracy_variance = np.var(fold_accuracies, ddof=1)

print(f"K-Fold Cross-Validation Mean Accuracy: {mean_accuracy}")
print(f"Accuracy Variance: {accuracy_variance}")
