# machine_learning\config_ml.py

# created 2/15/25
# last updated 2/15/25

from pathlib import Path
from package_setup_ml import PackageSetupML

script_filepath = Path(__file__).resolve()
package_setup = PackageSetupML(script_filepath, level="debug")
logger = package_setup.setup_logger()
scan_directory_path = package_setup.scan_directory_path