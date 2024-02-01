import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import roc_auc_score

from shared.data_handling import load_pregenerated_memmap_dataset
from shared.data_processing import flatten_generated_flow_pairs, flatten_arrays

def run_lazypredict(pregenerated_dataset_path):

    flow_pairs_train, labels_train, flow_pairs_test, labels_test = load_pregenerated_memmap_dataset(pregenerated_dataset_path)

    # Flatten the data. Created a 2D array from the 4D array. e.g. (1000, 8, 100, 1) -> (1000, 800)
    flattend_flow_pairs_train, flattend_flow_pairs_test = flatten_generated_flow_pairs(flow_pairs_train, flow_pairs_test)
    
    # Flatten the labels. Created a 1D array from the 2D array. e.g. (1000, 1) -> (1000,)
    flattend_labels_train, flattend_labels_test = flatten_arrays(labels_train, labels_test)

    # Initialize LazyClassifier
    clf = LazyClassifier(verbose=100, ignore_warnings=True, custom_metric=roc_auc_score)
    models, predictions = clf.fit(flattend_flow_pairs_train, flattend_flow_pairs_test, flattend_labels_train, flattend_labels_test)

    # Output the performance results
    print(models)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run LazyPredict on a pregenerated dataset')
    parser.add_argument('-pd','--pregenerated_dataset_path', type=str, required=True, help='Path to the pregenerated dataset')

    # Parse arguments
    args = parser.parse_args()

    # Run LazyPredict
    run_lazypredict(args.pregenerated_dataset_path)
