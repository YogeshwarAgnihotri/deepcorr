import pickle
import numpy as np
import os

def calc_train_test_indexes(dataset, train_ratio):
    print("\nSplitting dataset into training and testing sets...")
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

def save_test_indexes_to_path(test_indexes, path):
    print("Saving testing indexes to path...")
    # Create path for testing indexes
    test_indexes_path = os.path.join(path, "testing_indexes.pickle")

    # Save the testing indexes for later testing
    with open(test_indexes_path, 'wb') as f:
        pickle.dump(test_indexes, f)