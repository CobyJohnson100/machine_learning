# util_ml.py

from config_ml import scan_directory_path, logger

def ingest_datasets():
    import json
    datasets_filepath = scan_directory_path / "ingest" / "datasets.json"
    
    with open(datasets_filepath, 'r') as file:
        datasets = json.load(file)
    
    return datasets