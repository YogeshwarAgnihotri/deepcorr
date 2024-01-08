def prepare_data_for_training(flow_pairs_train, labels_train, flow_pairs_test, labels_test):
    flow_pairs_train = flatten_data(flow_pairs_train)  # Flatten the data for feeding into a decision tree
    flow_pairs_test = flatten_data(flow_pairs_test)  # Flatten the data for feeding into a decision tree

    labels_train = labels_train.flatten()  # Flatten the labels for feeding into a decision tree
    labels_test = labels_test.flatten()  # Flatten the labels for feeding into a decision tree
    
    return flow_pairs_train, labels_train, flow_pairs_test, labels_test

def flatten_data(data):
    """
    Flatten the data so it can be fed into a decision tree.

    :param data: Multidimensional data to be flattened.
    :return: Flattened data.
    """
    return data.reshape(data.shape[0], -1)