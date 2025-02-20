# machine_learning\LinearRegression.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config_ml import scan_directory_path, logger
from evalution_metric import calculate_evaluation_metrics

class LinearRegression:
    def __init__(self, dataset_filepath):
        self.dataset_filepath = dataset_filepath

    def singlevariate_linear_regression(self, feature_column, target_column):
        logger.info("Singlevariate Linear Regression")
        df = pd.read_csv(self.dataset_filepath)

        # Flatten the arrays to 1D
        ary_feature = df[feature_column].values.flatten()
        ary_target = df[target_column].values.flatten()

        ary_feature_train, ary_feature_test, ary_target_train, ary_target_test = train_test_split(
            ary_feature, 
            ary_target, 
            test_size=0.2, 
            random_state=42
        )
        
        feature_mean_train = np.mean(ary_feature_train)
        target_mean_train = np.mean(ary_target_train)

        # Calculate slope (m)
        numerator = np.sum((ary_feature - feature_mean_train) * (ary_target - target_mean_train))
        denominator = np.sum((ary_feature - feature_mean_train) ** 2)
        slope = numerator / denominator

        # Calculate intercept (b)
        intercept = target_mean_train - slope * feature_mean_train

        # Make predictions on test set
        y_prediction = slope * ary_feature_test + intercept

        # Calculate metrics using actual test targets
        evaluation_metrics = calculate_evaluation_metrics(ary_target_test, y_prediction)

        logger.info(f"Feature mean: {feature_mean_train:.4f}")
        logger.info(f"Target mean: {target_mean_train:.4f}")
        logger.info(f"Slope (m): {slope:.4f}")
        logger.info(f"Intercept (b): {intercept:.4f}")
        logger.info(str(evaluation_metrics))

        return y_prediction, evaluation_metrics, slope, intercept
    
    def multivariate_linear_regression(self, feature_columns, target_column):
        logger.info("Mutivariate Linear Regression")
        df = pd.read_csv(self.dataset_filepath)
        
        X = df[feature_columns].values
        y = df[target_column].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_mean = np.mean(X_train, axis=0)
        y_mean = np.mean(y_train)

        # Center the features
        X_centered = X_train - X_mean
        y_centered = y_train - y_mean

        # Calculate coefficients using normal equation: β = (X^T X)^(-1) X^T y
        coefficients = np.linalg.inv(X_centered.T @ X_centered) @ X_centered.T @ y_centered
        intercept = y_mean - np.sum(coefficients * X_mean)

        # Make predictions on test set
        y_prediction = X_test @ coefficients + intercept

        evaluation_metrics = calculate_evaluation_metrics(y_test, y_prediction)

        logger.info("Model Parameters:")
        for feature, coef in zip(feature_columns, coefficients):
            logger.info(f"{feature} coefficient: {coef:.4f}")
        logger.info(f"Intercept: {intercept:.4f}")
        logger.info(str(evaluation_metrics))

        return y_prediction, evaluation_metrics, coefficients, intercept

def linear_regression_main(dataset, single):
    import util_ml
    datasets = util_ml.ingest_datasets()

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
    
    linear_regression = LinearRegression(dataset_filepath)
    if single:
        y_prediction, evaluation_metrics, _, _ = linear_regression.singlevariate_linear_regression(
            'RM',
            'MEDV'
        )
    else:
        y_prediction, evaluation_metrics, _, _ =linear_regression.multivariate_linear_regression(
            ['RM', 'ZN', 'INDUS', 'CHAS', 'AGE', 'LSTAT', 'CRIM', 'RAD', 'PTRATIO', 'B', 'NOX', 'DIS', 'TAX'], 
            'MEDV'
        )

if __name__ == "__main__":
    import os, datetime, time, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--single', action='store_true', help='Run singlevariate linear regression (default: multivariate)')
    parser.add_argument('--dataset', type=str, default='boston', choices=['boston'], help='Dataset to use (default: boston)')
    args = parser.parse_args()

    script_name = os.path.basename(__file__)
    logger.info(f"Run {script_name} directly")

    start_time = time.time()
    linear_regression_main(args.dataset, args.single)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_readable = str(datetime.timedelta(seconds=execution_time))
    logger.info(f"Execution time: {execution_time_readable}")
