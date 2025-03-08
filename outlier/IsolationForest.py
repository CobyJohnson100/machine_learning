# machine_learning/outlier/IsolationForest.py

import numpy as np
from typing import Optional, List, Tuple
import random

class IsolationTree:
    def __init__(self, height_limit: int):
        self.height_limit = height_limit
        self.split_feature = None
        self.split_value = None
        self.size = 0
        self.left = None
        self.right = None
        self.height = 0
        
    def fit(self, X: np.ndarray, current_height: int) -> None:
        self.size = len(X)
        
        if current_height >= self.height_limit or self.size <= 1:
            self.height = current_height
            return
            
        n_features = X.shape[1]
        self.split_feature = random.randint(0, n_features - 1)
        
        min_val = X[:, self.split_feature].min()
        max_val = X[:, self.split_feature].max()
        
        if min_val == max_val:
            self.height = current_height
            return
            
        self.split_value = random.uniform(min_val, max_val)
        
        left_indices = X[:, self.split_feature] < self.split_value
        X_left = X[left_indices]
        X_right = X[~left_indices]
        
        if len(X_left) > 0:
            self.left = IsolationTree(self.height_limit)
            self.left.fit(X_left, current_height + 1)
            
        if len(X_right) > 0:
            self.right = IsolationTree(self.height_limit)
            self.right.fit(X_right, current_height + 1)
            
    def path_length(self, x: np.ndarray) -> float:
        if self.left is None and self.right is None:
            return self.height
            
        if x[self.split_feature] < self.split_value:
            if self.left is None:
                return self.height
            return self.left.path_length(x)
        else:
            if self.right is None:
                return self.height
            return self.right.path_length(x)

class IsolationForest:
    def __init__(self, n_estimators: int = 100, max_samples: str = 'auto', 
                 contamination: float = 0.1, random_state: Optional[int] = None):
        """
        Initialize Isolation Forest for anomaly detection.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of isolation trees to build
        max_samples : str or int, default='auto'
            Number of samples to draw to train each tree
        contamination : float, default=0.1
            Expected proportion of outliers in the dataset
        random_state : int, optional
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.trees: List[IsolationTree] = []
        self.threshold = 0.0
        
    def _average_path_length(self, n: int) -> float:
        """Calculate average path length given sample size."""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        
    def fit(self, X: np.ndarray) -> 'IsolationForest':
        """
        Fit the Isolation Forest model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        
        Returns:
        --------
        self : object
            Fitted estimator
        """
        if isinstance(X, list):
            X = np.array(X)
            
        n_samples = X.shape[0]
        
        if self.max_samples == 'auto':
            max_samples = min(256, n_samples)
        else:
            max_samples = min(self.max_samples, n_samples)
            
        height_limit = int(np.ceil(np.log2(max_samples)))
        
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            
        self.trees = []
        for _ in range(self.n_estimators):
            sample_idx = np.random.choice(n_samples, size=max_samples, replace=False)
            X_sample = X[sample_idx]
            
            tree = IsolationTree(height_limit)
            tree.fit(X_sample, 0)
            self.trees.append(tree)
            
        # Calculate threshold
        scores = self.score_samples(X)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
            
        return self
        
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores for samples.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        scores : ndarray
            Anomaly scores of input samples
        """
        if isinstance(X, list):
            X = np.array(X)
            
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            paths = np.array([tree.path_length(X[i]) for tree in self.trees])
            scores[i] = -np.mean(paths) / self._average_path_length(len(self.trees))
            
        return scores
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict if samples are outliers or not.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        predictions : ndarray
            1 for inliers, -1 for outliers
        """
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold, 1, -1)
        
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and predict if samples are outliers or not.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        predictions : ndarray
            1 for inliers, -1 for outliers
        """
        return self.fit(X).predict(X)
