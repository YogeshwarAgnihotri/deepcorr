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
    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(**config['decision_tree'])
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**config['random_forest'])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Hyperparameter search settings
    hyperparameter_search_config = config['hyperparameter_search']
    search_type = hyperparameter_search_config['type']

    if search_type != 'none':
        parameter_grid_path = hyperparameter_search_config['param_grid_path']
        if parameter_grid_path:
            parameter_grid = load_yaml(parameter_grid_path)
        else:
            raise ValueError("Parameter grid path must be specified for hyperparameter search.")

        if search_type == 'grid_search':
            model, _ = train_classifier_gridSearch(
                model, flow_pairs_train, labels_train, parameter_grid, hyperparameter_search_config['cross_validation_folds'], 
                hyperparameter_search_config['search_verbosity']
            )
        elif search_type == 'random_search':
            model, _ = train_classifier_randomSearch(
                model, flow_pairs_train, labels_train, parameter_grid, hyperparameter_search_config['random_search']['n_iter'], 
                hyperparameter_search_config['cross_validation_folds'], verbosity=hyperparameter_search_config['search_verbosity']
            )
    else:
        model = train_model(model, flow_pairs_train, labels_train)

    # Make predictions and evaluate the model
    predictions = model.predict(flow_pairs_test)
    evaluate_model_print_metrics(true_labels=labels_test, predicted_labels=predictions)

    # Print all configurations used for this run
    print(f"Configuration used: {config}")

if __name__ == "__main__":
    main()
