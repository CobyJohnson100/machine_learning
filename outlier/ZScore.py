# machine_learning/outlier/ZScore
from typing import List, Dict
import pandas as pd
import numpy as np

from config_ml import scan_directory_path, logger
import util_ml

class ZScore:
    def __init__(self):
        pass

    def calculate_column_zscore(
        self, 
        df: pd.DataFrame, 
        column: str
    ) -> pd.Series:
        mean = df[column].mean()
        std = df[column].std()
        
        # Avoid division by zero
        if std == 0:
            return pd.Series([0] * len(df), index=df.index)
            
        z_scores = (df[column] - mean) / std
        return z_scores

    def identify_outliers(
        self, 
        df: pd.DataFrame, 
        column: str
    ) -> pd.DataFrame:
        z_scores = self.calculate_column_zscore(df, column)
        
        # Find rows where absolute z-score exceeds threshold
        outlier_mask = abs(z_scores) > self.threshold
        outliers_df = df[outlier_mask].copy()
        
        # Add z-score column to output
        outliers_df['z_score'] = z_scores[outlier_mask]
        
        logger.info(f"Found {len(outliers_df)} outliers in column '{column}' "
                   f"with |z-score| > {self.threshold}")
        return outliers_df

    def identify_outliers_all_columns(
        self, 
        df: pd.DataFrame, 
        exclude_columns: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        if exclude_columns is None:
            exclude_columns = []
            
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers_dict = {}
        
        for column in numeric_columns:
            if column not in exclude_columns:
                outliers = self.identify_outliers(df, column)
                if len(outliers) > 0:
                    outliers_dict[column] = outliers
                    
        return outliers_dict

def zscore_main(dataset):
    ingest_datasets = util_ml.ingest_datasets()

    if dataset == 'boston':
        from ingest.IngestBostonHousing import IngestBostonHousing
        ingest_boston_housing = IngestBostonHousing()
        ingest_boston_housing.load_boston_housing_dataset()
        dataset_filepath = ingest_boston_housing.boston_housing_filepath
        column_of_interest = 'MEDV'
    elif dataset == 'iris':
        from ingest.IngestIris import IngestIris
        ingest_iris = IngestIris()
        ingest_iris.load_iris_dataset()
        dataset_filepath = ingest_iris.iris_filepath
    
    df = pd.read_csv(dataset_filepath)
    zscore = ZScore()
    z_score = zscore.calculate_column_zscore(df, ingest_datasets[dataset]['target'])
    logger.info(f"Z-score for {column_of_interest}:\n{z_score}")
    
if __name__ == "__main__":
    import os, datetime, time, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='boston', choices=['boston', 'iris'], help='Dataset to use (default: boston)')
    args = parser.parse_args()

    script_name = os.path.basename(__file__)
    logger.info(f"Run {script_name} directly")

    start_time = time.time()
    zscore_main(args.dataset)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_readable = str(datetime.timedelta(seconds=execution_time))
    logger.info(f"Execution time: {execution_time_readable}")
