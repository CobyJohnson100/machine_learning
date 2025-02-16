# machine_learning\package_setup_ml.py

# established 2/15/25
# last updated 2/15/25

import logging
from pathlib import Path

class PackageSetupML:
    def __init__(self, script_filepath, level="info", scan_directory="machine_learning"):
        self.script_filepath = script_filepath
        self.level = level
        self.scan_directory = scan_directory
        self.scan_directory_path = self._get_scan_directory_path()

    ### DIRECTORY ###
    def _get_scan_directory_path(self):
        script_path = Path(self.script_filepath).resolve()
        for part in script_path.parents:
            if part.name == self.scan_directory:
                return part
        raise ValueError(f"Scan directory {self.scan_directory} not found in {self.script_filepath}")

    ### LOGGING ###
    def setup_logger(self, logger_name=None):
        logger_name = logger_name or self.scan_directory
        logger = logging.getLogger(logger_name)

        if logger.hasHandlers():
            return logger

        log_directory = self.scan_directory_path / "log"
        if not log_directory.exists():
            log_directory.mkdir(parents=True)

        log_path = log_directory / f"{logger_name}.log"
        with open(log_path, 'w'):
            pass

        if self.level == "info":
            logging_level = logging.INFO
        elif self.level == "debug":
            logging_level = logging.DEBUG
        else:
            raise ValueError(f"Invalid logging level: {self.level}")
        logger.setLevel(logging_level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        stream_handler_exists = any(type(handler) is logging.StreamHandler for handler in logger.handlers)

        if not stream_handler_exists:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        logger.info(f"Logging initialized for {logger_name}")
        
        return logger