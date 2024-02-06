from xgboost import XGBClassifier
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def config_checks_hyperparameter_tuning(config):    
    if ((config['hyperparameter_search_strategy'] == 'none')
        or (config['selected_hyperparameter_grid'] == 'none')):
        raise ValueError(f"Config Setting Error: \
                         hyperparameter_search_strategy and \
                         selected_hyperparameter_grid must be set to a value.")
    
def config_checks_training(config):
    # Check if 'runs' is empty
    runs = config.get('runs', {})

    if not runs:
        raise ValueError("Config Setting Error: 'runs' is empty.")

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
    
def init_model_for_training(model_type, model_params):
    if model_type == 'decision_tree':
        return DecisionTreeClassifier(**model_params)
    elif model_type == 'random_forest':
        return RandomForestClassifier(**model_params)
    elif model_type == 'xgbClassifier':
        return XGBClassifier(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
def init_model_hyperparameter_tuning(model_type):
    if model_type == 'decision_tree':
        return DecisionTreeClassifier()
    elif model_type == 'random_forest':
        return RandomForestClassifier()
    elif model_type == 'xgbClassifier':
        return XGBClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")