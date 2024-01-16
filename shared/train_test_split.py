import pickle
import numpy as np
import os

def calc_train_test_indexes_using_ratio(dataset, train_ratio):
    print(f"\nSplitting dataset into training and testing sets using train_ratio {train_ratio}...")
    len_tr = len(dataset)

    rr = list(range(len(dataset)))
    np.random.shuffle(rr)

    train_indexes = rr[:int(len_tr*train_ratio)]
    print('Dataset length/Amount of true flow pairs for TRAINING: ', \
          len(train_indexes), "true flow pairs")

    test_indexes = rr[int(len_tr*train_ratio):]
    print("Dataset length/Amount of true flow pairs for TESTING: ", \
          len(test_indexes), "true flow pairs")
    
    return train_indexes, test_indexes

def calc_train_test_index_manual_split(dataset, training_true_flow_amount, testing_true_flow_amount):
    print(f"\nSplitting dataset into training and testing sets using manual split. Training true flow amount {training_true_flow_amount}/Testing True Flow amount {testing_true_flow_amount}...")
    len_tr = len(dataset)

    rr = list(range(len_tr))
    np.random.shuffle(rr)

    # Ensure that the sum of train_limit and test_limit doesn't exceed the dataset size
    if training_true_flow_amount + testing_true_flow_amount > len_tr:
        raise ValueError("Combined training and testing limits exceed the dataset size.")

    train_indexes = rr[:training_true_flow_amount]
    print('Dataset length/Amount of true flow pairs for TRAINING: ', len(train_indexes), "true flow pairs")

    # Select indices for the test set right after the training set indices
    test_indexes = rr[training_true_flow_amount:training_true_flow_amount + testing_true_flow_amount]
    print("Dataset length/Amount of true flow pairs for TESTING: ", len(test_indexes), "true flow pairs")

    return train_indexes, test_indexes

def save_test_indexes_to_path(test_indexes, path):
    print("Saving testing indexes to path...")
    # Create path for testing indexes
    test_indexes_path = os.path.join(path, "testing_indexes.pickle")

    # Save the testing indexes for later testing
    with open(test_indexes_path, 'wb') as f:
        pickle.dump(test_indexes, f)