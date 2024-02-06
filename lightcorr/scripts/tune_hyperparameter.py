# Needed to find the other modules. Dont really like this solution.
import sys
import os
sys.path.insert(0, 
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, 
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import argparse
import pandas
import time

from modules.model_persistence import save_model
from modules.model_training import (
    train_classifier_halvingGridSearch,
    train_classifier_gridSearch, 
    train_classifier_randomSearch
)
from modules.model_validation import perform_validation
from modules.config_utlis import init_model_hyperparameter_tuning
from modules.enviroment_setup import setup_environment
from modules.data_handling import load_prepare_dataset, save_dataset_info
from modules.data_processing import flatten_flow_pairs_and_label

from shared.utils import export_dataframe_to_csv
from shared.utils import copy_file


def perform_hyperparameter_search(search_strategy,
                                   model, 
                                   X_train, 
                                   y_train, 
                                   parameter_grid, 
                                   config, 
                                   output_path):
    # Retrieve the specific search configuration
    search_config = config['hyperparameter_search_settings'][search_strategy]
    
    # Map of search type to function
    search_functions = {
        'grid_search': train_classifier_gridSearch,
        'halving_grid_search': train_classifier_halvingGridSearch,
        'random_search': train_classifier_randomSearch
    }
    
    if search_strategy in search_functions:
        best_model, best_hyperparameters, cv_results = (
            search_functions[search_strategy](
            model, X_train, y_train, parameter_grid, **search_config))
        
        # Export results to CSV
        results_df = pandas.DataFrame(cv_results)
        export_dataframe_to_csv(results_df, 
                                f"{search_strategy}_cv_results.csv", 
                                output_path)
    else:
        raise ValueError(f"Unsupported hyperparameter search\
                          type: {search_strategy}")
    
    return best_model, best_hyperparameters

def main():
    """Tune hyperparameters for a Classifier using a dataset."""
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description='Train a Classifier on the dataset.'
    )
    parser.add_argument(
        '-c', '--config_path', type=str, required=True, 
        help='Path to the configuration file'
    )
    parser.add_argument(
        '-r', '--run_name', type=str, 
        help='Name of the run followed by date time. \
            If not set the current date and time only will be used.'
            
    )
    args = parser.parse_args()

    config, run_folder_path = setup_environment(args)

    print("\nModel type:", config['model_type'])
    print("Hyperparameter search type:", 
          config['hyperparameter_search_strategy'])
    print("Selected hyperparameter grid:",
          config['selected_hyperparameter_grid'])
    
    copy_file(args.config_path, os.path.join(
        run_folder_path, "used_config_hyperparameter_tune.yaml"))

    pregenerated_dataset_path = config['pregenerated_dataset_path']
    load_dataset_into_memory = config['load_data_set_into_memory']

    flow_pairs_train, labels_train, flow_pairs_test, labels_test = \
        load_prepare_dataset(pregenerated_dataset_path, 
                             load_dataset_into_memory)

    flattened_flow_pairs_train, flattened_labels_train = \
        flatten_flow_pairs_and_label(flow_pairs_train, labels_train)

    # Model initialization
    model_type = config['model_type']
    model = init_model_hyperparameter_tuning(model_type)

    hyperparameter_search_strategy = config['hyperparameter_search_strategy']

    # Dynamically select the hyperparameter grid
    if hyperparameter_search_strategy != 'none':
        selected_hyperparameter_grid = config['selected_hyperparameter_grid']
        parameter_grid = config['hyperparameter_grid'][model_type].get(
            selected_hyperparameter_grid, {})

    # Hyperparameter search or training
    if hyperparameter_search_strategy != 'none':
        best_model, best_hyperparameters = \
            perform_hyperparameter_search(hyperparameter_search_strategy, 
                                          model, 
                                          flattened_flow_pairs_train,
                                          flattened_labels_train, 
                                          parameter_grid, 
                                          config, 
                                          run_folder_path)
    else:
        raise ValueError("Hyperparameter search type is 'none' or not \
                         defined.")
    
    perform_validation(best_model, 
                       flattened_flow_pairs_train, 
                       flattened_labels_train, 
                       config, 
                       run_folder_path)

    save_model(best_model, run_folder_path)

    save_dataset_info(config, 
                      flow_pairs_train, 
                      labels_train, 
                      flow_pairs_test, 
                      labels_test, 
                      run_folder_path)

    end_time = time.time()
    print(f"\nFull training process finished in {end_time - start_time} seconds.")
    
if __name__ == "__main__":
    main()