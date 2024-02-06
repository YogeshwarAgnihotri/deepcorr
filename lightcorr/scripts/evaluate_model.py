# Needed to find the other modules. Dont really like this solution.
import sys
import os
sys.path.insert(0, 
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), '../modules')))
sys.path.insert(0, 
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), '../..')))

import joblib

from shared.model_evaluation import evaluate_test_set
from shared.data_handling import load_pregenerated_memmap_dataset
from shared.data_processing import flatten_generated_flow_pairs, flatten_arrays
from shared.utils import create_run_folder

def main():
    # parameter
    model_path = "/home/yogeshwar/master_thesis_corr/lightcorr/runs/04-02-2024_17:58:02_rf__default_agg/model.joblib"
    pregenerated_dataset_path = "/home/yogeshwar/datasets/deepcorr_pregen/deepcorr_9008_2252_agg"
    run_folder_path = "/home/yogeshwar/master_thesis_corr/lightcorr"
    run_name = "test_rf_default_agg"

    run_folder_path = create_run_folder(run_folder_path, run_name)

    model = joblib.load(model_path)

    flow_pairs_train, labels_train, flow_pairs_test, labels_test = \
        load_pregenerated_memmap_dataset(pregenerated_dataset_path)
    
    flattend_flow_pairs_test = flatten_generated_flow_pairs(flow_pairs_test)[0]

    flattend_labels_test = flatten_arrays(labels_test)[0]

    # Evaluate the model on the test set
    evaluate_test_set(model, 
                      flattend_flow_pairs_test, 
                      flattend_labels_test, 
                      run_folder_path)

if __name__ == "__main__":
    main()