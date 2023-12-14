import pickle
import tqdm
from utils import check_path_throw_error
import os

def load_dataset(path_dataset, load_only_min_300_flows=True):
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
        if load_only_min_300_flows:
            paths_to_pickle_files_to_load.append(
                os.path.join(path_dataset, f"{name}_tordata300.pickle"))
        else:
            paths_to_pickle_files_to_load.extend([
                os.path.join(path_dataset, f"{name}_tordata.pickle"),
                os.path.join(path_dataset, f"{name}_tordata300.pickle"),
                os.path.join(path_dataset, f"{name}_tordata400.pickle"),
                os.path.join(path_dataset, f"{name}_tordata500.pickle")
            ])

    with tqdm.tqdm(total=len(paths_to_pickle_files_to_load),
                   desc="Loading dataset from pickle files") as pbar:
        for path in paths_to_pickle_files_to_load:
            with open(path, 'rb') as file:
                dataset += pickle.load(file)
            pbar.update(1)

    return dataset

def load_test_index():
    with open('test_index300.pickle', 'rb') as file:
        test_index = pickle.load(file)[:1000]
    return test_index