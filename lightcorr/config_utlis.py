import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def config_checks(config):
    if config['hyperparameter_search_type'] != 'none' and config['single_model_training_config'] != 'none':
        raise ValueError(f"Config Setting Error: Single_model_training_config must be None when hyperparameter_search_type is not none. Cant search for hyperparamers and traning a single model at the same time.")
    if config['hyperparameter_search_type'] == 'none' and config['single_model_training_config'] == 'none':
        raise ValueError(f"Config Setting Error: single_model_training_config and hyperparameter_search_type are both none. Cant do nothing.")
    if (config['hyperparameter_search_type'] != 'none' and config['selected_hyperparameter_grid'] == 'none') or (config['hyperparameter_search_type'] == 'none' and config['selected_hyperparameter_grid'] != 'none'):
        raise ValueError("Config Setting Error: Both hyperparameter_search_type and selected_hyperparameter_grid must be set to a value if hyperparamter search is wished, otherwise both must be set to none.")

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
    if model_type == 'random_forest':
        if search_type == 'none':
            single_model_training_config = config['single_model_training_config']
            # Use predefined parameters for single model training
            model_params = config['single_model_training']['random_forest'][single_model_training_config]
            return RandomForestClassifier(**model_params)
        elif search_type == 'grid_search':
            # Initialize without parameters for hyperparameter search
            return RandomForestClassifier()
        elif search_type == 'random_search':
            # Initialize without parameters for hyperparameter search
            return RandomForestClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")