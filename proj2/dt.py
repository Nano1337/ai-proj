import pandas as pd
import numpy as np
import math 

#%%
'''
The Node class creates a node in the decision tree. It has the following attributes: 
- feature: the feature to split on (if it is a non-leaf node)
- threshold: the threshold to split on (if it is a non-leaf node)
- left: the left child node
- right: the right child node
- value: the prediction value (if it is a leaf node)
- counts: the count of each class in the node (used for classification)
'''
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

'''
The DecisionTree class creates a decision tree. It has the following attributes:
- max_depth: the maximum depth of the tree
- root: the root node of the tree
- class_dict: a dictionary to map the class index to the class name
'''
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self.class_dict = {0: 'average', 1: 'good', 2: 'great', 3: 'low', 4: 'very low'}

    '''
    The fit method trains the decision tree on the given data. 
    It creates the root node and calls the _grow_tree method to grow the tree.
    It accepts the following parameters:
    - X: the input features as a pandas DataFrame
    - y: the target labels as a pandas Series
    '''
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
        self.print_tree()

    '''
    The print_tree method prints the decision tree (used for debugging). 
    '''
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

    '''
    predict method predicts the target labels for the given input features.
    It calls the _traverse_tree method to traverse the tree and make predictions.
    It accepts the following parameters:
    - X: the input features as a pandas DataFrame
    '''
    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X.values]
    
    '''
    The _grow_tree method grows the decision tree. 
    It does this by recursively splitting the data based on the best split at each node until max dpeth is reached.
    It accepts the following parameters:
    - X: the input features as a pandas DataFrame
    - y: the target labels as a pandas Series
    '''
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

    '''
    The calculate_entropy method calculates the entropy of the given target labels.
    Entropy is a measure of impurity in the data. Impurity is 0 when all samples at a node belong to the same class.
    It accepts the following parameters:
    - y: the target labels as a pandas Series
    '''
    def calculate_entropy(self, y):
        if len(y) == 0:  # To avoid log(0)
            return 0
        proportions = np.array([sum(y == i) for i in np.unique(y)]) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    '''
    The information_gain method calculates the information gain of a split.
    Information gain is the difference in entropy between the parent node and the weighted average of the child nodes.
    It accepts the following parameters:
    - X_column: the input feature as a pandas Series
    - y: the target labels as a pandas Series
    - threshold: the threshold to split on
    '''
    def information_gain(self, X_column, y, threshold):
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

    '''
    The _best_split method finds the best split for a node.
    It does this by iterating through each feature and threshold to find the split with the highest information gain.
    It accepts the following parameters:
    - X: the input features as a pandas DataFrame
    - y: the target labels as a pandas Series
    '''
    def _best_split(self, X, y):
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
            
    '''
    The _traverse_tree method traverses the decision tree to make predictions.
    It does this by recursively traversing the tree based on the input features until a leaf node is reached.
    It accepts the following parameters:
    - x: the input features as a pandas Series
    - node: the current node in the decision tree
    '''
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

