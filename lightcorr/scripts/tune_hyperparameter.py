import argparse
import shutil
import joblib
import numpy as np
import time

# Needed to find the other modules. Dont really like this solution.
import sys
import os
sys.path.insert(0, 
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), '../modules')))
sys.path.insert(0, 
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), '../..')))

import pandas

from model_training import (
    train_classifier_halvingGridSearch,
    train_classifier_gridSearch, 
    train_classifier_randomSearch
)

from model_validation import (
    custom_cross_validate,
)

from config_utlis import (
    config_checks_hyperparameter_tuning, 
    load_config, 
    init_model_hyperparameter_tuning,
)

from shared.utils import (
    StreamToLogger, 
    setup_logger, 
    create_run_folder, export_dataframe_to_csv,
)

from shared.data_processing import (
    generate_flow_pairs_to_memmap, 
    flatten_arrays, 
    flatten_generated_flow_pairs,
)

from shared.train_test_split import (
    calc_train_test_indexes_using_ratio,
)

from shared.data_handling import (
    load_dataset_deepcorr, 
    load_pregenerated_memmap_dataset, 
    save_memmap_info_flow_pairs_labels,
)

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

    config = load_config(args.config_path)
    config_checks_hyperparameter_tuning(config)

    run_folder_path = create_run_folder(config['run_folder_path'], 
                                        args.run_name)
    output_file_path = os.path.join(run_folder_path, "training_log.txt")
    logger = setup_logger('TrainingLogger', output_file_path)
    sys.stdout = StreamToLogger(logger, sys.stdout)

    print("\nModel type:", config['model_type'])
    print("Hyperparameter search type:", 
          config['hyperparameter_search_strategy'])
    print("Selected hyperparameter grid:",
          config['selected_hyperparameter_grid'])
    
    config_file_destination = os.path.join(
        run_folder_path, "used_config_hyperparameter_tune.yaml"
    )
    shutil.copy(args.config_path, config_file_destination)

    if config['load_pregenerated_dataset']:
        # Load pregenerated dataset
        pregenerated_dataset_path = config['pregenerated_dataset_path']
        flow_pairs_train, labels_train, flow_pairs_test, labels_test = (
            load_pregenerated_memmap_dataset(pregenerated_dataset_path)
        )
    else:
        # Generate own dataset for this run
        # Extract settings from config
        dataset_path = config['base_dataset_path']
        train_ratio = config['train_ratio']
        flow_size = config['flow_size']
        negative_samples = config['negative_samples']
        load_all_data = config['load_all_data']
        memmap_dataset_path = os.path.join(run_folder_path, 'memmap_dataset')

        # Load dataset
        deepcorr_dataset = load_dataset_deepcorr(dataset_path, load_all_data)
        train_indexes, test_indexes = calc_train_test_indexes_using_ratio(
            deepcorr_dataset, train_ratio
        )
        flow_pairs_train, labels_train, flow_pairs_test, labels_test = (
            generate_flow_pairs_to_memmap(
                dataset=deepcorr_dataset, 
                train_index=train_indexes, 
                test_index=test_indexes, 
                flow_size=flow_size, 
                memmap_saving_path=memmap_dataset_path, 
                negative_samples=negative_samples
            )
        )

    # Doesn't seem to make any difference in speed,
    # maybe because the dataset is small.
    # Leaving it in for now.
    if config['load_dataset_into_memory']:
        flow_pairs_train, labels_train, flow_pairs_test, labels_test = [
            np.array(arr) for arr in (
                flow_pairs_train, labels_train, flow_pairs_test, labels_test
            )
        ]

    # Flatten the data. Created a 2D array from the 4D array.
    # e.g., (1000, 8, 100, 1) -> (1000, 800)
    flattened_flow_pairs_train = (
        flatten_generated_flow_pairs(flow_pairs_train)[0])

    # Flatten the labels. Created a 1D array from the 2D array.
    # e.g., (1000, 1) -> (1000,)
    flattened_labels_train = flatten_arrays(labels_train)[0]

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
    
    if config['validation_settings']['run_validation']:
        # Validate the model on the training set with cross validation
        validation_config = config['validation_settings']['cross_validation']
        roc_plot_enabled = config['validation_settings']['roc_plot_enabled']
        custom_cross_validate(best_model, 
                              flattened_flow_pairs_train, 
                              flattened_labels_train,
                              roc_plot_enabled, 
                              run_folder_path, 
                              **validation_config)

    # Save model for later evaluation or prediction making
    joblib.dump(best_model, os.path.join(run_folder_path, 'model.joblib'))

    if not config['load_pregenerated_dataset']:
        # In this case, the dataset is generated by this script and not loaded
        # from a pregenerated memmap file. Therefore, save the used dataset.
        print("\nSaving generated dataset to run folder...")
        save_memmap_info_flow_pairs_labels(flow_pairs_train, 
                                           labels_train, 
                                           flow_pairs_test, 
                                           labels_test, 
                                           run_folder_path)

    end_time = time.time()
    print(f"\nFull training process finished in \
          {end_time - start_time} seconds.")
    
if __name__ == "__main__":
    main()