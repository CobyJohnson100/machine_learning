# machine_learning/data_exploration/Plotter.py
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from config_ml import scan_directory_path, logger
import util_ml

class Plotter:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def plot_correlation_matrix(self):
        """Plot correlation matrix heatmap"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Boston Housing Features')
        plt.tight_layout()
        plt.show()
        
    def plot_distribution(self, column: str):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=column, kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()
        
    def plot_scatter_matrix(self, features: List[str], target: str):
        """Plot scatter plots for selected features against domain"""
        
        n_features = len(features)
        n_rows = int(np.ceil(n_features / 2)) 

        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(features):
            sns.scatterplot(data=self.df, x=feature, y=target, ax=axes[idx])
            axes[idx].set_title(f'{feature} vs {target}')
        
        plt.tight_layout()
        plt.show()
        
    def plot_boxplots(self):
        """Plot boxplots for all features"""
        plt.figure(figsize=(15, 6))
        self.df.boxplot()
        plt.xticks(rotation=45)
        plt.title('Boxplots of Boston Housing Features')
        plt.tight_layout()
        plt.show()

def plotter_main(dataset, plot_type):
    ingest_datasets = util_ml.ingest_datasets()

    if dataset == 'boston':
        from ingest.IngestBostonHousing import IngestBostonHousing
        ingest_boston_housing = IngestBostonHousing()
        ingest_boston_housing.load_boston_housing_dataset()
        dataset_filepath = ingest_boston_housing.boston_housing_filepath
        dataset_name = "boston"
    elif dataset == 'iris':
        from ingest.IngestIris import IngestIris
        ingest_iris = IngestIris()
        ingest_iris.load_iris_dataset()
        dataset_filepath = ingest_iris.iris_filepath
        dataset_name = "iris"

    features = ingest_datasets[dataset_name]['multi_features']
    target = ingest_datasets[dataset_name]['target']

    df = pd.read_csv(dataset_filepath)
        
    plotter = Plotter(df)
    
    if plot_type == 'correlation':
        plotter.plot_correlation_matrix()
    elif plot_type == 'distribution':
        plotter.plot_distribution(target)
    elif plot_type == 'scatter':
        plotter.plot_scatter_matrix(features, target)
    elif plot_type == 'boxplot':
        plotter.plot_boxplots()
    elif plot_type == 'all':
        plotter.plot_correlation_matrix()
        plotter.plot_distribution(target)
        plotter.plot_scatter_matrix(features, target)
        plotter.plot_boxplots()

if __name__ == "__main__":
    import os, datetime, time, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_type', type=str, default='all', 
                      choices=['correlation', 'distribution', 'scatter', 'boxplot', 'all'],
                      help='Type of plot to generate (default: all)')
    parser.add_argument('--dataset', type=str, default='boston', 
                      choices=['boston', 'iris'], 
                      help='Dataset to use (default: boston)')
    args = parser.parse_args()

    script_name = os.path.basename(__file__)
    logger.info(f"Run {script_name} directly")

    start_time = time.time()
    plotter_main(args.dataset, args.plot_type)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_readable = str(datetime.timedelta(seconds=execution_time))
    logger.info(f"Execution time: {execution_time_readable}")