# machine_learning/ingest/IngestIris.py
import pandas as pd

from config_ml import scan_directory_path, logger

class IngestIris:
    def __init__(self):
        self.iris_filepath = scan_directory_path / "working" / "iris_dataset.csv"

    def load_iris_dataset(self):
        from sklearn.datasets import load_iris
        iris = load_iris()

        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name='target')
        
        # Add target names as a separate column
        target_names = pd.Series([iris.target_names[i] for i in iris.target], name='species')
        
        # Combine features and target
        df = pd.concat([X, y, target_names], axis=1)
        
        logger.info(f"Iris dataset loaded with {df.shape[0]} samples and {df.shape[1]} features")
        logger.info(f"Features: {iris.feature_names}")
        logger.info(f"Target classes: {iris.target_names}")
        logger.info(f"First 3 rows of the dataset:\n{df.head(3)}")

        df.to_csv(self.iris_filepath, index=False)
        return df

def ingest_iris_dataset():
    ingest_iris = IngestIris()
    ingest_iris.load_iris_dataset()

if __name__ == "__main__":
    import os, datetime, time

    script_name = os.path.basename(__file__)
    logger.info(f"Run {script_name} directly")
    
    start_time = time.time()
    ingest_iris_dataset()
    end_time = time.time()
    
    execution_time = end_time - start_time
    execution_time_readable = str(datetime.timedelta(seconds=execution_time))
    logger.info(f"Execution time: {execution_time_readable}")