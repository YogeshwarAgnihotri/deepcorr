from shared.data_processing import (
    flatten_arrays, 
    flatten_generated_flow_pairs,
)

def flatten_flow_pairs_and_label(flow_pairs, labels):
    # Flatten the data. Created a 2D array from the 4D array.
    # e.g., (1000, 8, 100, 1) -> (1000, 800)
    flattened_flow_pairs = flatten_generated_flow_pairs(flow_pairs)[0]

    # Flatten the labels. Created a 1D array from the 2D array.
    # e.g., (1000, 1) -> (1000,)
    flattened_labels = flatten_arrays(labels)[0]

    return flattened_flow_pairs, flattened_labels