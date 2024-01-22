import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas
import argparse
import shutil


import yaml

from model_training import train_model, train_classifier_gridSearch, train_classifier_randomSearch
from model_evaluation import custom_cross_validate, evaluate_test_set
from config_utlis import config_checks, load_config, initialize_model

from shared.utils import StreamToLogger, setup_logger, create_run_folder, export_dataframe_to_csv
from shared.data_processing import generate_flow_pairs_to_memmap, flatten_arrays, flatten_generated_flow_pairs
from shared.train_test_split import calc_train_test_indexes_using_ratio
from shared.data_handling import load_dataset_deepcorr, load_pregenerated_memmap_dataset

def main():
    # Parse only the config file path
    parser = argparse.ArgumentParser(description='Train a Classifier on the dataset.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--run_name', type=str, help='Name of the run followed by date time. If not set the current date and time only will be used.')
    args = parser.parse_args() 

    # Load configuration
    config = load_config(args.config_path)
    config_checks(config)

    # Print the configuration (only idenityfing information)
    print("\nModel type:", config['model_type'])
    print("Hyperparameter search type:", config['hyperparameter_search_type'])
    print("Hyperparameter grid:", config['selected_hyperparameter_grid'])
    print("Single model training config:", config['single_model_training_config'])

    # Prepare the run folder and logger
    run_folder_path = config['run_folder_path']
    run_folder_path = create_run_folder(run_folder_path, args.run_name)
    output_file_path = os.path.join(run_folder_path, "training_log.txt")
    logger = setup_logger('TrainingLogger', output_file_path)
    sys.stdout = StreamToLogger(logger, sys.stdout)

    # Copy the configuration file to the run folder and rename it to "used_config.yaml"
    config_file_destination = os.path.join(run_folder_path, "used_config.yaml")
    shutil.copy(args.config_path, config_file_destination)

    
    if config['load_pregenerated_dataset']:
        # Load pregenerated dataset
        pregenerated_dataset_path = config['pregenerated_dataset_path']
        flow_pairs_train, labels_train, flow_pairs_test, labels_test = load_pregenerated_memmap_dataset(pregenerated_dataset_path)
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
        train_indexes, test_indexes = calc_train_test_indexes_using_ratio(deepcorr_dataset, train_ratio)
        flow_pairs_train, labels_train, flow_pairs_test, labels_test = generate_flow_pairs_to_memmap(
            dataset=deepcorr_dataset, 
            train_index=train_indexes, 
            test_index=test_indexes, 
            flow_size=flow_size, 
            memmap_saving_path=memmap_dataset_path, 
            negetive_samples=negative_samples)
        
    # for debugging TODO remove later
    #export_dataframe_to_csv(pandas.DataFrame(flow_pairs_train), 'flow_pairs_train_before_flattening.csv', run_folder_path)

    # Flatten the data. Created a 2D array from the 4D array. e.g. (1000, 8, 100, 1) -> (1000, 800)
    flattend_flow_pairs_train, flattend_flow_pairs_test = flatten_generated_flow_pairs(flow_pairs_train, flow_pairs_test)
    
    # Flatten the labels. Created a 1D array from the 2D array. e.g. (1000, 1) -> (1000,)
    flattend_labels_train, flattend_labels_test = flatten_arrays(labels_train, labels_test)
    
    # for debugging TODO remove later
    #export_dataframe_to_csv(pandas.DataFrame(flow_pairs_train), 'flow_pairs_train.csv', run_folder_path)

    # Model initialization
    model_type = config['model_type']
    hyperparameter_search_type = config['hyperparameter_search_type']
    model = initialize_model(config, model_type, hyperparameter_search_type)

    # Dynamically select the hyperparameter grid
    if hyperparameter_search_type != 'none':
        selected_hyperparameter_grid = config['selected_hyperparameter_grid']
        parameter_grid = config['hyperparameter_grid'][model_type].get(selected_hyperparameter_grid, {})

    # Hyperparameter search or training
    if hyperparameter_search_type == 'grid_search':
        grid_search_config = config['hyperparameter_search_settings']['grid_search']
        best_model, best_hyperparameters, cv_results = train_classifier_gridSearch(
            model, flattend_flow_pairs_train, flattend_labels_train, parameter_grid, **grid_search_config
        ) 
        export_dataframe_to_csv(pandas.DataFrame(cv_results), 'grid_search_cv_results.csv', run_folder_path)
    elif hyperparameter_search_type == 'random_search':
        random_search_config = config['hyperparameter_search_settings']['random_search']
        best_model, best_hyperparameters, cv_results = train_classifier_randomSearch(
            model, flattend_flow_pairs_train, flattend_labels_train, parameter_grid, **random_search_config
        )
        export_dataframe_to_csv(pandas.DataFrame(cv_results), 'random_search_cv_results.csv', run_folder_path)
    elif hyperparameter_search_type == 'none':
        # this is just a model, not the best model :)
        best_model = train_model(model, flattend_flow_pairs_train, flattend_labels_train)
    else:
        raise ValueError(f"Unsupported hyperparameter search type: {hyperparameter_search_type}")

    # Evaluate the model on the training set with cross validation
    validation_config = config['validation_settings']['cross_validation']
    roc_plot_enabled = config['validation_settings']['roc_plot_enabled']
    custom_cross_validate(best_model, flattend_flow_pairs_train, flattend_labels_train, roc_plot_enabled, run_folder_path, **validation_config)

    if config['evaluation_settings']['evaluate_on_test_set']:
        # Evaluate the model on the test set
        evaluate_test_set(best_model, flattend_flow_pairs_test, flattend_labels_test)


if __name__ == "__main__":
    main()