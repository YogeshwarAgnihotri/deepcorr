import pickle
import tqdm


def load_dataset(path_to_dataset, load_only_min_300_flows=True):
    all_runs={'8872':'192.168.122.117','8802':'192.168.122.117','8873':'192.168.122.67','8803':'192.168.122.67',
            '8874':'192.168.122.113','8804':'192.168.122.113','8875':'192.168.122.120',
            '8876':'192.168.122.30','8877':'192.168.122.208','8878':'192.168.122.58'}

    dataset=[]

    #build all paths for better loading bar
    paths_to_pickle_files_to_load=[]
    for name in all_runs:
        if load_only_min_300_flows:
            paths_to_pickle_files_to_load.append(path_to_dataset + '/%s_tordata300.pickle' % name)
        else:
            paths_to_pickle_files_to_load.append(path_to_dataset + '/%s_tordata.pickle' % name)
            paths_to_pickle_files_to_load.append(path_to_dataset + '/%s_tordata300.pickle' % name)
            paths_to_pickle_files_to_load.append(path_to_dataset + '/%s_tordata400.pickle' % name)
            paths_to_pickle_files_to_load.append(path_to_dataset + '/%s_tordata500.pickle' % name)

    with tqdm.tqdm(total=len(paths_to_pickle_files_to_load), desc="Loading dataset from pickle files") as pbar:
        for path in paths_to_pickle_files_to_load:
            dataset += pickle.load(open(path))
            pbar.update(1)

    return dataset


def load_test_index():
    test_index=pickle.load(open('test_index300.pickle'))[:1000]
    return test_index