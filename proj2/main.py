import pandas as pd
import numpy as np
import math 

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [sum(y == i) for i in range(1, 6)]
        predicted_class = num_samples_per_class.index(max(num_samples_per_class)) + 1
        node = Node(value=predicted_class)

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        # Get threshold for each feature, equivalent to the mean of the feature 
        thresholds = [np.mean(X[:, feature]) for feature in range(X.shape[1])]

        # Calculate score for each feature 
        score = []
        for t in range(len(thresholds)):
            thresh = thresholds[t] 
            p1 = sum(X[:, t] < thresh) / X.shape[0]
            p2 = sum(X[:, t] >= thresh) / X.shape[0]
            p1_inner = 0 
            p2_inner = 0
            for i in range(1, 6):
                p1_inner += sum(y[X[:, t] < thresh] == i) / sum(X[:, t] < thresh) * math.log(sum(y[X[:, t] < thresh] == i) / sum(X[:, t] < thresh))
                p2_inner += sum(y[X[:, t] >= thresh] == i) / sum(X[:, t] >= thresh) * math.log(sum(y[X[:, t] >= thresh] == i) / sum(X[:, t] >= thresh))
            p1 = p1 * p1_inner
            p2 = p2 * p2_inner
            score.append(p1 + p2) 
        return np.argmin(score), thresholds[np.argmin(score)]
            

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
# Preprocess data
def utility_to_label(utility):
    utility = float(utility)
    if utility <= 0.2:
        return 'very low'
    elif utility <= 0.4:
        return 'low'
    elif utility <= 0.6:
        return 'average'
    elif utility <= 0.8:
        return 'good'
    else:
        return 'great'

df = pd.read_csv('data.txt', sep='\t', encoding='utf-16')
df['utility'] = df['utility'].apply(utility_to_label)
df.to_csv('data.txt', sep='\t', encoding='utf-16', index=False)

