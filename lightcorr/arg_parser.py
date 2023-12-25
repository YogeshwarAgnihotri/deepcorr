import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a Decision Tree Classifier on DeepCorr dataset.')
    # data parameters
    parser.add_argument('--dataset_path', type=str, default="/home/yagnihotri/datasets/deepcorr_original_dataset", help='Path to the DeepCorr dataset (default: %(default)s)')
    parser.add_argument('--run_folder_path', type=str, default="/home/yagnihotri/projects/corr/treecorr", help='Path to the folder where the run folder will be created (default: %(default)s)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio (default: %(default)s)')
    parser.add_argument('--flow_size', type=int, default=300, help='Flow size (default: %(default)s)')
    parser.add_argument('--negative_samples', type=int, default=199, help='Number of negative samples (default: %(default)s)')
    parser.add_argument('--load_all_data', action='store_true', help='If NOT set, only flows with minimum 300 packets will be loaded (about 7300 flow pairs). If set, all will be loaded (about 20-25k).')

    #### training parameters
    # Decision Tree parameters for a single tree to train
    parser.add_argument('--criterion', type=str, default="gini", help='The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain (default: %(default)s)')
    parser.add_argument('--splitter', type=str, default="best", help='The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split. (default: %(default)s)')
    parser.add_argument('--max_depth', type=int, default=None, help='The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. (default: %(default)s)')
    parser.add_argument('--min_samples_split', type=int, default=2, help='The minimum number of samples required to split an internal node (default: %(default)s)')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='The minimum number of samples required to be at a leaf node (default: %(default)s)')
    parser.add_argument('--min_weight_fraction_leaf', type=float, default=0.0, help='The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided. (default: %(default)s)')
    parser.add_argument('--max_features', type=parse_max_features, default=None, help='The number of features to consider when looking for the best split. Accepts an integer, a float in (0.0, 1.0], or one of {"sqrt", "log2", "auto"}. (default: %(default)s)')
    parser.add_argument('--random_state', type=int, default=None, help='Controls the randomness of the estimator. The features are always randomly permuted at each split, even if splitter is set to “best”. When max_features < n_features, the algorithm will select max_features at random at each split before finding the best split among them. But the best found split may vary across different runs, even if max_features=n_features. That is the case, if the improvement of the criterion is identical for several splits and one split has to be selected at random. To obtain a deterministic behaviour during fitting, random_state has to be fixed to an integer. (default: %(default)s)')
    parser.add_argument('--max_leaf_nodes', type=int, default=None, help='Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes. (default: %(default)s)')
    parser.add_argument('--min_impurity_decrease', type=float, default=0.0, help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value. (default: %(default)s)')
    parser.add_argument('--class_weight', type=str, default=None, help='Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. (default: %(default)s)')
    parser.add_argument('--ccp_alpha', type=float, default=0.0, help='Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. (default: %(default)s)')

    #parameters to set for grid search of tree hyperparameters
    parser.add_argument('--search_verbosity', type=int, default=2, help='Controls the verbosity: the higher, the more messages. (default: %(default)s)')
    parser.add_argument('--cross_validation_folds', type=int, default=5, help='Number of folds for cross validation (default: %(default)s)')
    parser.add_argument('--tree_param_path', type=str, help='Path to JSON file containing parameter grid')

    parser.add_argument('--do_tree_grid_search', action='store_true', help='If set, grid hyperparameter search will be done for the decision tree.')
    
    parser.add_argument('--do_tree_random_search', action='store_true', help='If set, random hyperparameter search will be done for the decision tree.')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution. (default: %(default)s)')



    return parser.parse_args()

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
        return str(value)

    # If none of the above, raise an error
    raise argparse.ArgumentTypeError(f"Invalid value for --max_features: {value}")
