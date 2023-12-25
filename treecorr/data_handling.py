from shared.data_loader import load_dataset_deepcorr
from shared.data_processing import generate_flow_pairs
from shared.train_test_split import calc_train_test_indexes

def load_data(dataset_path, load_all_data=False):
    """
    Load the dataset from the specified path.

    :param dataset_path: Path to the dataset.
    :param load_all_data: Flag to determine whether to load all data or a subset.
    :return: Loaded dataset.
    """
    return load_dataset_deepcorr(dataset_path, load_all_data)

def prepare_data_for_dt_training(deepcorr_dataset, train_ratio, flow_size, run_folder_path, negative_samples):
    """
    Prepare the dataset for training and testing. This involves splitting the dataset 
    and generating flow pairs and labels.

    :param deepcorr_dataset: The loaded dataset.
    :param train_ratio: Ratio to split the dataset into training and testing.
    :param flow_size: The size of each flow in the dataset.
    :param run_folder_path: Path where any output should be saved.
    :param negative_samples: Number of negative samples to include.
    :return: Tuple of training data and labels, testing data and labels.
    """
    train_indexes, test_indexes = calc_train_test_indexes(deepcorr_dataset, train_ratio)
    flow_pairs_train, labels_train, flow_pairs_test, labels_test = generate_flow_pairs(
        deepcorr_dataset, train_indexes, test_indexes, flow_size, run_folder_path, negative_samples)
    
    flow_pairs_train = flatten_data_for_decision_tree(flow_pairs_train)  # Flatten the data for feeding into a decision tree
    flow_pairs_test = flatten_data_for_decision_tree(flow_pairs_test)  # Flatten the data for feeding into a decision tree
    
    return (flow_pairs_train, labels_train), (flow_pairs_test, labels_test)

def flatten_data_for_decision_tree(data):
    """
    Flatten the data so it can be fed into a decision tree.

    :param data: Multidimensional data to be flattened.
    :return: Flattened data.
    """
    return data.reshape(data.shape[0], -1)