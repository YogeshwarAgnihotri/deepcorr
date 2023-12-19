import argparse
import datetime
import logging
import sys
import os
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GridSearchCV


from shared.utils import create_path
from shared.data_loader import load_dataset_deepcorr
from shared.data_processing import generate_flow_pairs 
from shared.train_test_split import calc_train_test_indexes

import time

#################### Classes and Functions ####################
class StreamToLogger:
    """
    Custom stream object that redirects writes to both a logger and the original stdout.
    """
    def __init__(self, logger, orig_stdout):
        self.logger = logger
        self.orig_stdout = orig_stdout
        self.linebuf = ''  # Buffer to accumulate lines until a newline character

    def write(self, message):
        """
        Write the message to logger and the original stdout.
        Only log messages when a newline character is encountered.
        """
        self.linebuf += message
        if '\n' in message:
            self.flush()

    def flush(self):
        """Flush the stream by logging the accumulated line and clearing the buffer."""
        if self.linebuf.rstrip():
            self.logger.info(self.linebuf.rstrip())
        self.orig_stdout.write(self.linebuf)
        self.linebuf = ''

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger."""
    formatter = logging.Formatter('%(message)s')  # Only include the message in logs

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    return logger

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def format_time(seconds):
    """Format time in seconds to days, hours, minutes, and seconds."""
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s"

def flatten_data_for_decision_tree(data):
    """Flatten the data for feeding into a decision tree."""
    return data.reshape(data.shape[0], -1)

def evaluate_model_print_metrics(true_labels, predicted_labels):
    # Evaluate the model
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Print the evaluation results
    print(f'\nAccuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    # Assuming conf_matrix is the result of confusion_matrix(labels_test, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()

    print("Confusion Matrix:")
    print(f"{'':>10} {'Predicted':>18}")
    print(f"{'':>10} {'0':>8} {'1':>8}")
    print(f"{'Actual 0':>10} {TN:>8} {FP:>8}")
    print(f"{'Actual 1':>10} {FN:>8} {TP:>8}")

    # Calculate True Positive Rate (TPR) also known as Recall
    TPR = TP / (TP + FN)  # TPR = Recall
    # Calculate False Positive Rate (FPR)
    FPR = FP / (FP + TN)
    print(f"True Positive Rate (TPR/Recall): {TPR:.2f}")
    print(f"False Positive Rate (FPR): {FPR:.2f}")

def dt_train_classifier_gridSearch(param_grid, cross_validation_folds, verbosity, training_data, labels):
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=cross_validation_folds, scoring='accuracy', verbose=verbosity, n_jobs=-1)

    # Start timing the training process
    start_time_grid_search = time.time()

    # do grid seach and make predictions with best model
    print("\nTraining the model with grid search of tree hyperparameters...")
    grid_search.fit(training_data, labels)
    best_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_
    # End timing the training process with grid search
    training_time_grid_search = time.time() - start_time_grid_search
    print(f"Training completed in {format_time(training_time_grid_search)}")

    print(f"Best parameters: {best_parameters}")

    return best_model, best_parameters

def dt_train(training_data, 
             labels,
             criterion="gini", 
             splitter="best", 
             max_depth=None, 
             min_samples_split=2, 
             min_samples_leaf=1, 
             min_weight_fraction_leaf=0.0, 
             max_features=None, 
             random_state=None, 
             max_leaf_nodes=None, 
             min_impurity_decrease=0.0, 
             class_weight=None, 
             ccp_alpha=0.0):

    # Create a Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion=criterion, 
                                 splitter=splitter, 
                                 max_depth=max_depth, 
                                 min_samples_split=min_samples_split, 
                                 min_samples_leaf=min_samples_leaf, 
                                 min_weight_fraction_leaf=min_weight_fraction_leaf, 
                                 max_features=max_features, 
                                 random_state=random_state, 
                                 max_leaf_nodes=max_leaf_nodes,
                                 min_impurity_decrease=min_impurity_decrease, 
                                 class_weight=class_weight, 
                                 ccp_alpha=ccp_alpha)
    # Start timing the training process
    start_time_traning = time.time()
    print("\nTraining the decision tree...")
    # Train the model
    model = clf.fit(training_data, labels)
    # End timing the training process
    training_time = time.time() - start_time_traning
    print(f"Training of decision tree completed in {format_time(training_time)}")

    return model

def parse_max_features(value):
    # Attempt to parse as an integer
    try:
        return int(value)
    except ValueError:
        pass

    # Attempt to parse as a float
    try:
        float_value = float(value)
        if 0.0 < float_value <= 1.0:
            return float_value
    except ValueError:
        pass

    # Check if it's a valid string option
    if value in ['sqrt', 'log2', 'auto']:
        return value

    # If none of the above, raise an error
    raise argparse.ArgumentTypeError(f"Invalid value for --max_features: {value}")


#################### Parameters ####################
# Set up argument parser
parser = argparse.ArgumentParser(description='Train a Decision Tree Classifier on DeepCorr dataset.')
# data parameters
parser.add_argument('--dataset_path', type=str, default="/home/yagnihotri/datasets/deepcorr_original_dataset", help='Path to the DeepCorr dataset')
parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
parser.add_argument('--flow_size', type=int, default=300, help='Flow size')
parser.add_argument('--negative_samples', type=int, default=199, help='Number of negative samples')
parser.add_argument('--load_all_data', action='store_true', help='If NOT set, only flows with minimum 300 packets will be loaded (about 7300 flow pairs). If set, all will be loaded (about 20-25k).')

#### training parameters
# Decision Tree parameters for a single tree to train
parser.add_argument('--criterion', type=str, default="gini", help='The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain')
parser.add_argument('--splitter', type=str, default="best", help='The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.')
parser.add_argument('--max_depth', type=int, default=None, help='The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.')
parser.add_argument('--min_samples_split', type=int, default=2, help='The minimum number of samples required to split an internal node')
parser.add_argument('--min_samples_leaf', type=int, default=1, help='The minimum number of samples required to be at a leaf node')
parser.add_argument('--min_weight_fraction_leaf', type=float, default=0.0, help='The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.')
parser.add_argument('--max_features', type=parse_max_features, default=None, help='The number of features to consider when looking for the best split. Accepts an integer, a float in (0.0, 1.0], or one of {"sqrt", "log2", "auto"}.')
parser.add_argument('--random_state', type=int, default=None, help='Controls the randomness of the estimator. The features are always randomly permuted at each split, even if splitter is set to “best”. When max_features < n_features, the algorithm will select max_features at random at each split before finding the best split among them. But the best found split may vary across different runs, even if max_features=n_features. That is the case, if the improvement of the criterion is identical for several splits and one split has to be selected at random. To obtain a deterministic behaviour during fitting, random_state has to be fixed to an integer.')
parser.add_argument('--max_leaf_nodes', type=int, default=None, help='Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.')
parser.add_argument('--min_impurity_decrease', type=float, default=0.0, help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value.')
parser.add_argument('--class_weight', type=str, default=None, help='Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.')
parser.add_argument('--ccp_alpha', type=float, default=0.0, help='Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed.')

#parameters to set for grid search of tree hyperparameters
parser.add_argument('--do_tree_hyperparameter_search', action='store_true', help='If set, hyperparameter search will be done for the decision tree.')
parser.add_argument('--param_grid_path', type=str, help='Path to JSON file containing parameter grid')
parser.add_argument('--cross_validation_folds', type=int, default=5, help='Number of folds for cross validation')

# Parse arguments
args = parser.parse_args()

# Extracting arguments
dataset_path = args.dataset_path
train_ratio = args.train_ratio
flow_size = args.flow_size
negative_samples = args.negative_samples
load_all_data = args.load_all_data

criterion = args.criterion
splitter = args.splitter
max_depth = args.max_depth
min_samples_split = args.min_samples_split
min_samples_leaf = args.min_samples_leaf
min_weight_fraction_leaf = args.min_weight_fraction_leaf
max_features = args.max_features
random_state = args.random_state
max_leaf_nodes = args.max_leaf_nodes
min_impurity_decrease = args.min_impurity_decrease
class_weight = args.class_weight
ccp_alpha = args.ccp_alpha

do_tree_hyperparameter_search = args.do_tree_hyperparameter_search
param_grid_path = args.param_grid_path
if do_tree_hyperparameter_search:
    if param_grid_path is None:
        raise ValueError("Please set the path to the JSON file containing the parameter grid for grid search.")
    else:
        param_grid = load_yaml(param_grid_path)

cross_validation_folds = args.cross_validation_folds

#################### Path Stuff ####################
path_for_saving_run = "/home/yagnihotri/projects/corr/treecorr/runs"
# TODO Change run_name to something shorter up sometime. maybe with config files that show the full parameters
run_name = f"Date_{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
run_folder_path = os.path.join(path_for_saving_run, run_name)
create_path(run_folder_path)

output_file_path = os.path.join(run_folder_path, "training_log.txt")

logger = setup_logger('DeepCorrTraining', output_file_path)
# Redirect stdout
sys.stdout = StreamToLogger(logger, sys.stdout)

# Loading deepcorr dataset
deepcorr_dataset = load_dataset_deepcorr(dataset_path, load_all_data)
# Split the dataset into training and test sets
train_indexes, test_indexes = calc_train_test_indexes(deepcorr_dataset, train_ratio)
# Preprocess the data and generate the data arrays for training and testing
l2s, labels,l2s_test,labels_test = generate_flow_pairs(deepcorr_dataset, train_indexes, test_indexes, flow_size, run_folder_path, negative_samples)

l2s_flattened = flatten_data_for_decision_tree(l2s)  # Flatten the data for feeding into a decision tree

l2s_test_flattened = flatten_data_for_decision_tree(l2s_test)  # Flatten the data for feeding into a decision tree

if do_tree_hyperparameter_search:
    # do grid seach and make predictions with best model
    best_model, best_parameters = dt_train_classifier_gridSearch(param_grid, cross_validation_folds, 3, l2s_flattened, labels)
    predictions = best_model.predict(l2s_test_flattened)

else: # do normal training, with basic decision tree
    model = dt_train(l2s_flattened, 
                     labels, 
                     criterion, 
                     splitter, 
                     max_depth, 
                     min_samples_split, 
                     min_samples_leaf, 
                     min_weight_fraction_leaf, 
                     max_features, 
                     random_state, 
                     max_leaf_nodes, 
                     min_impurity_decrease, 
                     class_weight, 
                     ccp_alpha)
    predictions = model.predict(l2s_test_flattened)


evaluate_model_print_metrics(true_labels=labels_test, predicted_labels=predictions)

#print all arguments set by the user for this run
args_dict = vars(args)
print(args_dict)