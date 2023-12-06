import pickle
import numpy as np
from utlis import check_create_path, check_path_throw_error
import os



def split_train_test(dataset, train_ratio, run_folder_path):
    len_tr = len(dataset)
    print 'Dataset length: ', len_tr
    rr = range(len(dataset))
    np.random.shuffle(rr)

    train_index = rr[:int(len_tr*train_ratio)]
    print 'Length of true correlating flow pairs for TRAINING: ', \
          len(train_index), "flow pairs"

    test_index = rr[int(len_tr*train_ratio):]
    print "Length of true correlating flow pairs for TESTING: ", \
          len(test_index), "flow pairs"


    # Create path for testing indexes
    test_indexes_path = os.path.join(run_folder_path, "testing_indexes.pickle")

    # Save the testing indexes for later testing
    with open(test_indexes_path, 'wb') as f:
        pickle.dump(test_index, f)

    return train_index, test_index