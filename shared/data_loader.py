import pickle
import tqdm
from shared.utils import check_path_throw_error
import os
import numpy as np
import json 

def load_dataset_deepcorr(path_dataset, load_all_data=False):
    print("\nLoading dataset from base pickle files...")
    check_path_throw_error(path_dataset)

    runs = {
        '8872': '192.168.122.117',
        '8802': '192.168.122.117',
        '8873': '192.168.122.67',
        '8803': '192.168.122.67',
        '8874': '192.168.122.113',
        '8804': '192.168.122.113',
        '8875': '192.168.122.120',
        '8876': '192.168.122.30',
        '8877': '192.168.122.208',
        '8878': '192.168.122.58'
    }

    dataset = []

    # build all paths for better loading bar
    paths_to_pickle_files_to_load = []
    for name in runs:
        if load_all_data == False:
            paths_to_pickle_files_to_load.append(
                os.path.join(path_dataset, f"{name}_tordata300.pickle"))
        else:
            paths_to_pickle_files_to_load.extend([
                # TODO add the padding stuff from the other script (somewhere done already) and then use this. otherwise this doesent work currently
                # It contains about another 12.000 flow pairs (half of the full dataset)
                #os.path.join(path_dataset, f"{name}_tordata.pickle"),
                os.path.join(path_dataset, f"{name}_tordata300.pickle"),
                os.path.join(path_dataset, f"{name}_tordata400.pickle"),
                os.path.join(path_dataset, f"{name}_tordata500.pickle")
            ])

    with tqdm.tqdm(total=len(paths_to_pickle_files_to_load),
                   desc="Loading progress") as pbar:
        for path in paths_to_pickle_files_to_load:
            with open(path, 'rb') as file:
                dataset += pickle.load(file)
            pbar.update(1)

    len_tr = len(dataset)
    print('Dataset length/Amount of true flow pairs: ', len_tr)

    return dataset

def load_test_index_deepcorr():
    print("\nLoading testing indexes from pickle file...")
    with open('test_index300.pickle', 'rb') as file:
        test_index = pickle.load(file)[:1000]
    return test_index

def load_pregenerated_memmap_dataset(path):
    print("\nLoading pregenerated memmap dataset...")
    check_path_throw_error(path)

    # Load shapes from the JSON file
    shapes_file_path = os.path.join(path, 'memmap_shapes.json')
    with open(shapes_file_path, 'r') as f:
        shapes = json.load(f)

    labels_path = os.path.join(path, "training_labels")
    l2s_path = os.path.join(path, "training_flow_pairs")
    labels_test_path = os.path.join(path, "test_labels")
    l2s_test_path = os.path.join(path, "test_flow_pairs")

    labels = np.memmap(labels_path, dtype=np.float32, mode='r', shape=tuple(shapes["labels_train_shape"]))
    l2s = np.memmap(l2s_path, dtype=np.float32, mode='r', shape=tuple(shapes["flow_pairs_train_shape"]))
    labels_test = np.memmap(labels_test_path, dtype=np.float32, mode='r', shape=tuple(shapes["labels_test_shape"]))
    l2s_test = np.memmap(l2s_test_path, dtype=np.float32, mode='r', shape=tuple(shapes["flow_pairs_test_shape"]))

    # Count true and false labels
    true_training_pairs = np.sum(labels)
    false_training_pairs = labels.shape[0] - true_training_pairs
    true_testing_pairs = np.sum(labels_test)
    false_testing_pairs = labels_test.shape[0] - true_testing_pairs

    # Print the dataset sizes
    print("Pre-generated dataset loaded successfully! Dataset sizes:")
    print(f"TRAINING set size (true and false flow pairs total): {labels.shape[0]}")
    print(f"TRAINING set size of true flow pairs: {int(true_training_pairs)}")
    print(f"TRAINING set size of false flow pairs: {int(false_training_pairs)}")
    print(f"TESTING set size (true and false flow pairs total): {labels_test.shape[0]}")
    print(f"TESTING set size of true flow pairs: {int(true_testing_pairs)}")
    print(f"TESTING set size of false flow pairs: {int(false_testing_pairs)}")


    return l2s, labels, l2s_test, labels_test
