import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import shutil


import yaml

from sklearn.tree import DecisionTreeClassifier

from data_handling import prepare_data_for_dt_training
from model_training import train_model, train_classifier_gridSearch, train_classifier_randomSearch
from lightcorr.model_evaluation import evaluate_cross_val, evaluate_test_set

from shared.utils import StreamToLogger, setup_logger, create_run_folder
from shared.data_processing import generate_flow_pairs_to_memmap
from shared.train_test_split import calc_train_test_indexes
from shared.data_loader import load_dataset_deepcorr, load_pregenerated_memmap_dataset

def config_checks(config):
    if config['hyperparameter_search_type'] != 'none' and config['single_model_training_config'] != 'none':
        raise ValueError(f"single_model_training_config must be None when hyperparameter_search_type is not none. Cant search for hyperparamers and traning a single model at the same time.")
    if config['hyperparameter_search_type'] == 'none' and config['single_model_training_config'] == 'none':
        raise ValueError(f"single_model_training_config and hyperparameter_search_type are both none. Cant do nothing.")

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def initialize_model(config, model_type, search_type):
    if model_type == 'decision_tree':
        if search_type == 'none':
            single_model_training_config = config['single_model_training_config']
            # Use predefined parameters for single model training
            model_params = config['single_model_training']['decision_tree'][single_model_training_config]
            return DecisionTreeClassifier(**model_params)
        elif search_type == 'grid_search':
            # Initialize without parameters for hyperparameter search
            return DecisionTreeClassifier()
        elif search_type == 'random_search':
            # Initialize without parameters for hyperparameter search
            return DecisionTreeClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main():
    # Parse only the config file path
    parser = argparse.ArgumentParser(description='Train a Classifier on the dataset.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)
    config_checks(config)

    # Prepare the run folder and logger
    run_folder_path = config['run_folder_path']
    run_folder_path = create_run_folder(run_folder_path)
    output_file_path = os.path.join(run_folder_path, "training_log.txt")
    logger = setup_logger('TrainingLogger', output_file_path)
    sys.stdout = StreamToLogger(logger, sys.stdout)

    
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
        train_indexes, test_indexes = calc_train_test_indexes(deepcorr_dataset, train_ratio)
        flow_pairs_train, labels_train, flow_pairs_test, labels_test = generate_flow_pairs_to_memmap(
            dataset=deepcorr_dataset, 
            train_index=train_indexes, 
            test_index=test_indexes, 
            flow_size=flow_size, 
            memmap_saving_path=memmap_dataset_path, 
            negetive_samples=negative_samples)

    # some flattinening and stuff to make it work with the decision tree
    flow_pairs_train, labels_train, flow_pairs_test, labels_test = prepare_data_for_dt_training(
        flow_pairs_train, labels_train, flow_pairs_test, labels_test
    )

    # Model initialization
    model_type = config['model_type']
    hyperparameter_search_type = config['hyperparameter_search_type']
    model = initialize_model(config, model_type, hyperparameter_search_type)

    # Dynamically select the hyperparameter grid
    if hyperparameter_search_type != 'none':
        selected_hyperparameter_grid = config['selected_hyperparameter_grid']
        parameter_grid = config['hyperparameter_grid'].get(selected_hyperparameter_grid, {})

    # Hyperparameter search or training
    if hyperparameter_search_type == 'grid_search':
        grid_search_config = config['hyperparameter_search_settings']['grid_search']
        best_model, best_hyperparameters = train_classifier_gridSearch(
            model, flow_pairs_train, labels_train, parameter_grid, **grid_search_config
        )
    elif hyperparameter_search_type == 'random_search':
        random_search_config = config['hyperparameter_search_settings']['random_search']
        best_model, best_hyperparameters = train_classifier_randomSearch(
            model, flow_pairs_train, labels_train, parameter_grid, **random_search_config
        )
    elif hyperparameter_search_type == 'none':
        # this is just a model, not the best model :)
        best_model = train_model(model, flow_pairs_train, labels_train)
    else:
        raise ValueError(f"Unsupported hyperparameter search type: {hyperparameter_search_type}")

    # Evaluate the model on the training set with cross validation
    evalutation_config = config['evaluation_settings']['cross_validation']
    evaluate_cross_val(best_model, flow_pairs_train, labels_train, **evalutation_config)

    if config['evaluation_settings']['evaluate_on_test_set']:
        # Evaluate the model on the test set
        evaluate_test_set(best_model, flow_pairs_test, labels_test)

    # Copy the configuration file to the run folder and rename it to "used_config.yaml"
    config_file_destination = os.path.join(run_folder_path, "used_config.yaml")
    shutil.copy(args.config_path, config_file_destination)


if __name__ == "__main__":
    main()