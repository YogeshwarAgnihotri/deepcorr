import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from shared.utils import create_path, StreamToLogger, setup_logger, load_yaml, create_run_folder
from data_handling import load_data, prepare_data_for_dt_training
from model_training import train_model, train_classifier_gridSearch, train_classifier_randomSearch
from evaluation import evaluate_model_print_metrics

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def initialize_model(config, model_type, search_type):
    if model_type == 'decision_tree':
        if search_type == 'none':
            # Use predefined parameters for single model training
            model_params = config['single_model_training']['decision_tree']
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

    # Extract settings from config
    dataset_path = config['dataset_path']
    run_folder_path = config['run_folder_path']
    train_ratio = config['train_ratio']
    flow_size = config['flow_size']
    negative_samples = config['negative_samples']
    load_all_data = config['load_all_data']

    # Prepare the run folder and logger
    run_folder_path = create_run_folder(run_folder_path)
    output_file_path = os.path.join(run_folder_path, "training_log.txt")
    logger = setup_logger('TrainingLogger', output_file_path)
    sys.stdout = StreamToLogger(logger, sys.stdout)

    # Load dataset
    deepcorr_dataset = load_data(dataset_path, load_all_data)
    (flow_pairs_train, labels_train), (flow_pairs_test, labels_test) = prepare_data_for_dt_training(
        deepcorr_dataset, train_ratio, flow_size, run_folder_path, negative_samples
    )

    # Model initialization
    model_type = config['model_type']
    hyperparameter_search_type = config['hyperparameter_search_settings']['hyperparameter_search_type']
    model = initialize_model(config, model_type, hyperparameter_search_type)

    # Dynamically select the hyperparameter grid
    if hyperparameter_search_type != 'none':
        selected_hyperparameter_grid = config['hyperparameter_search_settings']['selected_hyperparameter_grid']
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
        best_model = train_model(model, flow_pairs_train, labels_train)
    else:
        raise ValueError(f"Unsupported hyperparameter search type: {hyperparameter_search_type}")


    # Make predictions and evaluate the model
    predictions = model.predict(flow_pairs_test)
    evaluate_model_print_metrics(true_labels=labels_test, predicted_labels=predictions)

    # Print all configurations used for this run
    print(f"Configuration used: {config}")

if __name__ == "__main__":
    main()