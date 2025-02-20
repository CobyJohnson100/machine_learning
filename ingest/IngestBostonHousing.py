# machine_learning/ingest/IngestBostonHousing.py
import pandas as pd

from config_ml import scan_directory_path, logger

class IngestBostonHousing:
    def __init__(self):
        self.boston_housing_filepath = scan_directory_path / "working" / "boston_housing_dataset.csv"

    def load_boston_housing_dataset(self):
        from sklearn.datasets import fetch_openml
        # Load Boston Housing dataset from OpenML
        boston = fetch_openml(name="boston", version=1, as_frame=True)
        X = boston.data
        y = boston.target
        
        # Convert to pandas DataFrame for easier handling
        df = pd.concat([X, y], axis=1)
        logger.info(f"Boston Housing dataset loaded with {df.shape[0]} samples and {df.shape[1]} features")
        logger.info(f"First 5 rows of the dataset:\n{df.head(3)}")

        df.to_csv(self.boston_housing_filepath, index=False)
        return df
    
def ingest_boston_housing_dataset():
    ingest_boston_housing = IngestBostonHousing()
    ingest_boston_housing.load_boston_housing_dataset()
    

if __name__ == "__main__":
    import os, datetime, time

    script_name = os.path.basename(__file__)
    logger.info(f"Run {script_name} directly")
    start_time = time.time()
    ingest_boston_housing_dataset()
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_readable = str(datetime.timedelta(seconds=execution_time))
    logger.info(f"Execution time: {execution_time_readable}")
