import sys
import os
import subprocess
import time
from pathlib import Path

def run_training(config_path):
    """
    Calls the train_model.py script with the given configuration file path.

    Parameters:
    - config_path (str): The path to the configuration file.
    """
    # Assuming train_model.py is in the same directory as this script
    script_directory = os.path.dirname(__file__)
    train_model_script_path = os.path.join(script_directory, 'train_model.py')
    
    # Prepare the command to call train_model.py with the configuration file
    command = ['python', train_model_script_path, '-c', config_path]
    
    # Execute the command
    subprocess.run(command, check=True)

def main():
    """Train classifiers on the dataset using different configuration files."""
    start_time = time.time()
    
    config_files = [
        "/home/yogeshwar/master_thesis_corr/lightcorr/scripts/train_model/train_model_config/evaluation/feature_dropped/dt.yaml",
        "/home/yogeshwar/master_thesis_corr/lightcorr/scripts/train_model/train_model_config/evaluation/feature_dropped/rf.yaml",
        "/home/yogeshwar/master_thesis_corr/lightcorr/scripts/train_model/train_model_config/evaluation/feature_dropped/xgb.yaml",
    ]

    for config_file in config_files:
        print(f"Starting training with configuration: {config_file}")
        run_training(config_file)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nFull training process for all models finished in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
