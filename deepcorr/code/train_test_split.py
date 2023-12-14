import pickle
import numpy as np
from utils import check_create_path, check_path_throw_error
import os

def split_train_test(dataset, train_ratio, run_folder_path):
    len_tr = len(dataset)
    print('Dataset length: ', len_tr)
    rr = list(range(len(dataset)))
    np.random.shuffle(rr)

    train_indexes = rr[:int(len_tr*train_ratio)]
    print('Length of true correlating flow pairs for TRAINING: ', \
          len(train_indexes), "flow pairs")

    test_indexes = rr[int(len_tr*train_ratio):]
    print("Length of true correlating flow pairs for TESTING: ", \
          len(test_indexes), "flow pairs")
    
    return train_indexes, test_indexes

def save_test_indexes_to_path(test_indexes, path):
    # Create path for testing indexes
    test_indexes_path = os.path.join(path, "testing_indexes.pickle")

    # Save the testing indexes for later testing
    with open(test_indexes_path, 'wb') as f:
        pickle.dump(test_indexes, f)