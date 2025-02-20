# machine_learning/GradientDescent.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config_ml import scan_directory_path, logger
from evalution_metric import calculate_evaluation_metrics

class GradientDescent:
    def __init__(self, dataset_filepath):
        self.dataset_filepath = dataset_filepath

    def singlevariate_gradient_descent(
        self, 
        feature_column, 
        target_column, 
        learning_rate=0.01, 
        epochs=1000, 
        batch_size=None
    ):
        df = pd.read_csv(self.dataset_filepath)
        X = df[feature_column].values.reshape(-1, 1)
        y = df[target_column].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize parameters
        m = 0  # slope
        b = 0  # intercept
        n_samples = len(X_train)
        
        # For plotting convergence
        loss_history = []

        for epoch in range(epochs):
            if batch_size:
                # Mini-batch gradient descent
                indices = np.random.permutation(n_samples)
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:min(i + batch_size, n_samples)]
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]
                    
                    # Calculate predictions
                    y_pred = m * X_batch + b
                    
                    # Calculate gradients
                    dm = (-2/len(X_batch)) * np.sum(X_batch * (y_batch - y_pred))
                    db = (-2/len(X_batch)) * np.sum(y_batch - y_pred)
                    
                    # Update parameters
                    m -= learning_rate * dm
                    b -= learning_rate * db
            else:
                # Batch gradient descent
                y_pred = m * X_train + b
                
                # Calculate gradients
                dm = (-2/n_samples) * np.sum(X_train * (y_train - y_pred))
                db = (-2/n_samples) * np.sum(y_train - y_pred)
                
                # Update parameters
                m -= learning_rate * dm
                b -= learning_rate * db
            
            # Calculate loss for monitoring
            y_pred = m * X_train + b
            mse = np.mean((y_train - y_pred) ** 2)
            loss_history.append(mse)
            
            # Early stopping (optional)
            if epoch > 0 and abs(loss_history[-1] - loss_history[-2]) < 1e-7:
                logger.info(f"Converged at epoch {epoch}")
                break

        # Make predictions on test set
        y_prediction = m * X_test + b
        metrics = calculate_evaluation_metrics(y_test, y_prediction)
        
        logger.info(f"Final Parameters (Gradient Descent):")
        logger.info(f"Slope (m): {m:.4f}")
        logger.info(f"Intercept (b): {b:.4f}")
        logger.info(str(metrics))
        
        return y_prediction, metrics, loss_history, m, b
    
    def multivariate_gradient_descent(
        self,
        feature_columns,
        target_columns,
        learning_rate=0.01,
        epochs=1000,
        batch_size=None
    ):
        df = pd.read_csv(self.dataset_filepath)
        
        # Select features and standardize them
        features = ['RM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT', 'CRIM', 
                    'RAD', 'PTRATIO', 'B', 'NOX', 'DIS', 'TAX']
        X = df[features].values
        y = df['MEDV'].values

        # Standardize features
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize parameters
        n_features = X_train.shape[1]
        weights = np.zeros(n_features)
        bias = 0
        n_samples = len(X_train)
        
        loss_history = []

        for epoch in range(epochs):
            if batch_size:
                # Mini-batch gradient descent
                indices = np.random.permutation(n_samples)
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:min(i + batch_size, n_samples)]
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]
                    
                    # Calculate predictions
                    y_pred = np.dot(X_batch, weights) + bias
                    
                    # Calculate gradients
                    dw = (-2/len(X_batch)) * np.dot(X_batch.T, (y_batch - y_pred))
                    db = (-2/len(X_batch)) * np.sum(y_batch - y_pred)
                    
                    # Update parameters
                    weights -= learning_rate * dw
                    bias -= learning_rate * db
            else:
                # Batch gradient descent
                y_pred = np.dot(X_train, weights) + bias
                
                # Calculate gradients
                dw = (-2/n_samples) * np.dot(X_train.T, (y_train - y_pred))
                db = (-2/n_samples) * np.sum(y_train - y_pred)
                
                # Update parameters
                weights -= learning_rate * dw
                bias -= learning_rate * db
            
            # Calculate loss for monitoring
            y_pred = np.dot(X_train, weights) + bias
            mse = np.mean((y_train - y_pred) ** 2)
            loss_history.append(mse)
            
            # Early stopping (optional)
            if epoch > 0 and abs(loss_history[-1] - loss_history[-2]) < 1e-7:
                logger.info(f"Converged at epoch {epoch}")
                break

        # Make predictions on test set
        y_prediction = np.dot(X_test, weights) + bias
        metrics = calculate_evaluation_metrics(y_test, y_prediction)
        
        logger.info("Final Parameters (Gradient Descent):")
        for feature, weight in zip(features, weights):
            logger.info(f"{feature} coefficient: {weight:.4f}")
        logger.info(f"Bias: {bias:.4f}")
        logger.info(str(metrics))
        
        return y_prediction, metrics, loss_history, weights, bias
    
def gradient_descent_main(dataset, single):
    if dataset == 'boston':
        from ingest.IngestBostonHousing import IngestBostonHousing
        ingest_boston_housing = IngestBostonHousing()
        ingest_boston_housing.load_boston_housing_dataset()
        dataset_filepath = ingest_boston_housing.boston_housing_filepath
    elif dataset == 'iris':
        from ingest.IngestIris import IngestIris
        ingest_iris = IngestIris()
        ingest_iris.load_iris_dataset()
        dataset_filepath = ingest_iris.iris_filepath
    
    gradient_descent = GradientDescent(dataset_filepath)
    if single:
        y_prediction, metrics, _, _, _ = gradient_descent.singlevariate_gradient_descent(
            'RM',
            'MEDV'
        )
    else:
        y_prediction, metrics, _, _, _ = gradient_descent.multivariate_gradient_descent(
            ['RM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT', 'CRIM', 'RAD', 'PTRATIO', 'B', 'NOX', 'DIS', 'TAX'], 
            'MEDV'
        )

if __name__ == "__main__":
    import os, datetime, time, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--single', action='store_true', help='Run singlevariate linear regression (default: multivariate)')
    parser.add_argument('--dataset', type=str, default='boston', choices=['boston', 'iris'], help='Dataset to use (default: boston)')
    args = parser.parse_args()

    script_name = os.path.basename(__file__)
    logger.info(f"Run {script_name} directly")
    start_time = time.time()
    gradient_descent_main(args.dataset, args.single)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_readable = str(datetime.timedelta(seconds=execution_time))
    logger.info(f"Execution time: {execution_time_readable}")
