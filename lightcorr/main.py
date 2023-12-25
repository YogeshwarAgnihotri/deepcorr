import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import datetime
from shared.utils import create_path, StreamToLogger, setup_logger, load_yaml, create_run_folder
from data_handling import load_data, prepare_data_for_dt_training, flatten_data_for_decision_tree
from model_training import dt_train, dt_train_classifier_gridSearch, dt_train_classifier_randomSearch
from evaluation import evaluate_model_print_metrics
from arg_parser import parse_arguments


def main():
    #################### Parameters ####################
    # Parse arguments
    args = parse_arguments()

    # Extracting arguments
    dataset_path = args.dataset_path
    run_folder_path = args.run_folder_path
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

    do_tree_grid_search = args.do_tree_grid_search
    do_tree_random_search = args.do_tree_random_search
    search_verbosity = args.search_verbosity
    tree_param_path = args.tree_param_path

    if do_tree_grid_search:
        if tree_param_path is None:
            raise ValueError("Please set the path to the JSON file containing the parameter grid for grid hyperparameter search.")
        else:
            param_grid = load_yaml(tree_param_path)

    if do_tree_random_search:
        if tree_param_path is None:
            raise ValueError("Please set the path to the JSON file containing the parameter grid for random hyperparameter search.")
        else:
            param_grid = load_yaml(tree_param_path)

    cross_validation_folds = args.cross_validation_folds
    n_iter = args.n_iter

    #################### Path Stuff ####################
    run_folder_path = create_run_folder(run_folder_path)
    output_file_path = os.path.join(run_folder_path, "training_log.txt")
    logger = setup_logger('DeepCorrTraining', output_file_path)
    # Redirect stdout
    sys.stdout = StreamToLogger(logger, sys.stdout)

    # Loading deepcorr dataset
    deepcorr_dataset = load_data(dataset_path, load_all_data)

    (flow_pairs_train, labels_train), (flow_pairs_test, labels_test) = prepare_data_for_dt_training(deepcorr_dataset, train_ratio, flow_size, run_folder_path, negative_samples)

    if do_tree_grid_search:
        # do grid seach and make predictions with best model
        best_model, best_parameters = dt_train_classifier_gridSearch(flow_pairs_train, labels_train, param_grid, cross_validation_folds, verbosity=search_verbosity)
        predictions = best_model.predict(flow_pairs_test)

    if do_tree_random_search:
        # do random seach and make predictions with best model
        best_model, best_parameters = dt_train_classifier_randomSearch(flow_pairs_train, labels_train, param_grid, n_iter, cross_validation_folds, verbosity=search_verbosity)
        predictions = best_model.predict(flow_pairs_test)

    else: # do normal training, with basic decision tree
        model = dt_train(flow_pairs_train, 
                        labels_train, 
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
        predictions = model.predict(flow_pairs_test)


    evaluate_model_print_metrics(true_labels=labels_test, predicted_labels=predictions)

    #print all arguments set by the user for this run
    args_dict = vars(args)
    print(args_dict)

if __name__ == "__main__":
    main()