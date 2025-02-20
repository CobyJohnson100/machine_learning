# machine_learning/LogisticRegression.py
import numpy as np

from config_ml import scan_directory_path, logger

class LogisticRegression:
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iteration: int = 1000,
        threshold: float = 0.5
    ):
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.threshold = threshold
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        """Sigmoid activation function
        
        Args:
            z: Input value or array
            
        Returns:
            Sigmoid of input: 1/(1 + e^(-z))
        """
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the logistic regression model using gradient descent
        
        Args:
            X: Training features (n_samples, n_features)
            y: Target values (n_samples,)
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            
            # Log progress every 100 iterations
            if (i + 1) % 100 == 0:
                loss = self._compute_loss(y, predictions)
                logger.info(f'Iteration {i+1}/{self.n_iterations}, Loss: {loss:.4f}')
    
    def predict_proba(self, X):
        """Predict probability of class 1
        
        Args:
            X: Features to predict (n_samples, n_features)
            
        Returns:
            Predicted probabilities (n_samples,)
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X):
        """Predict class labels
        
        Args:
            X: Features to predict (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)
    
    def _compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Average binary cross-entropy loss
        """
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
def logistic_regression_main(dataset):
    logistic_regression = LogisticRegression()

if __name__ == "__main__":
    import os, datetime, time, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='iris', choices=['iris'], help='Dataset to use (default: iris)')
    args = parser.parse_args()

    script_name = os.path.basename(__file__)
    logger.info(f"Run {script_name} directly")

    start_time = time.time()
    logistic_regression_main(args.dataset)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_readable = str(datetime.timedelta(seconds=execution_time))
    logger.info(f"Execution time: {execution_time_readable}")
