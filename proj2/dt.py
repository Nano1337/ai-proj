import pandas as pd
import numpy as np
import math 

#%%
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, counts=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.counts = counts
        self.is_leaf_node = False

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self.class_dict = {0: 'average', 1: 'good', 2: 'great', 3: 'low', 4: 'very low'}

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
        self.print_tree()
    
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        indent = "  " * depth
        if node.is_leaf_node:
            print(f"{indent}Leaf Node - Predict: {node.value} with counts: {node.counts}")
        else:
            print(f"{indent}Node - Feature: {node.feature}, Threshold: {node.threshold}, Counts: {node.counts}")
            if node.left is not None:
                print(f"{indent}Left:")
                self.print_tree(node.left, depth + 1)
            if node.right is not None:
                print(f"{indent}Right:")
                self.print_tree(node.right, depth + 1)


    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X.values]
        
    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [sum(y == i) for i in np.unique(y)]
        node = Node(value=np.unique(y)[np.argmax(num_samples_per_class)], counts=num_samples_per_class)
        self.print_tree(node, depth)
        if depth == self.max_depth:
            node.is_leaf_node = True

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X.iloc[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature = idx
                node.threshold = thr
                if X_left.shape[0] > 0:
                    node.left = self._grow_tree(X_left, y_left, depth + 1)
                if X_right.shape[0] > 0:
                    node.right = self._grow_tree(X_right, y_right, depth + 1)
                if node.left is None and node.right is None:
                    node.is_leaf_node = True
        return node

    def calculate_entropy(self, y):
        """Calculate the entropy of label distribution."""
        if len(y) == 0:  # To avoid log(0)
            return 0
        proportions = np.array([sum(y == i) for i in np.unique(y)]) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def information_gain(self, X_column, y, threshold):
        """Calculate information gain for a split on a given feature at a given threshold."""
        # Parent entropy
        parent_entropy = self.calculate_entropy(y)

        # Generate split
        left_mask = X_column < threshold
        right_mask = ~left_mask
        left_y = y[left_mask]
        right_y = y[right_mask]

        # Weighted average child entropy
        n = len(y)
        n_left, n_right = len(left_y), len(right_y)
        e_left, e_right = self.calculate_entropy(left_y), self.calculate_entropy(right_y)
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # Information gain
        ig = parent_entropy - child_entropy
        return ig

    def _best_split(self, X, y):
        """Find the best split for a node."""
        # Initialize variables to track the best split
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X.iloc[:, feature])
            
            for i in range(1, len(thresholds)):
                threshold = (thresholds[i - 1] + thresholds[i]) / 2
                ig = self.information_gain(X.iloc[:, feature], y, threshold)
                
                if ig > best_gain:
                    best_gain = ig
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold
            

    def _traverse_tree(self, x, node):
        try:
            if node.is_leaf_node or (node.left is None and node.right is None):
                return node.value
            if x[node.feature] < node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
        except:
            print(node.feature, node.threshold, node.left, node.right, node.is_leaf_node)
#%% 
    
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

