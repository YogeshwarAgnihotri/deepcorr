# quick cmd to run this script: 
""" 
python home/yagnihotri/projects/corr/scripts/pregenerate_dataset.py --dataset_path=/home/yagnihotri/datasets/deepcorr_original_dataset/ --save_directory=/home/yagnihotri/datasets/deepcorr_5000_1000_2 --negative_samples=1 --load_all_true_flow_pairs --true_flow_pairs_manual_split --true_flow_pairs_training_amount 5000 --true_flow_pairs_testing_amount 1000

"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json

from shared.utils import StreamToLogger, create_path, save_args_to_file, setup_logger, check_if_path_exists
from shared.data_handling import load_dataset_deepcorr
from shared.data_processing import generate_flow_pairs_to_memmap
from shared.train_test_split import calc_train_test_indexes_using_ratio, calc_train_test_index_manual_split

def main():
    parser = argparse.ArgumentParser(description='Split the deepcorr dataset into training and test indexes.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--save_directory', type=str, required=True, help='Directory to save the split indexes')
    parser.add_argument('--flow_size', type=int, default=300, help='Size of the flow')
    parser.add_argument('--negative_samples', type=int, default=1, help='Number of negative samples to include')
    
    parser.add_argument('--load_all_true_flow_pairs', action='store_true', help='Load all data if set')
    parser.add_argument('--true_flow_pairs_train_ratio', type=float, help='Training set ratio for automatic split')
    parser.add_argument('--true_flow_pairs_manual_split', action='store_true', help='Enable manual split')   
    parser.add_argument('--true_flow_pairs_training_amount', type=int, help='Number of true flow pairs to load for training in manual split')
    parser.add_argument('--true_flow_pairs_testing_amount', type=int, help='Number of true flow pairs to load for testing in manual split')

    args = parser.parse_args()

    if check_if_path_exists(args.save_directory):
        raise ValueError(f"Save directory {args.save_directory} already exists.")
    else:
        create_path(args.save_directory)
    
    logger = setup_logger('TrainingLogger', os.path.join(args.save_directory, "generation_console_output.txt"))
    sys.stdout = StreamToLogger(logger, sys.stdout)

    deepcorr_dataset = load_dataset_deepcorr(args.dataset_path, args.load_all_true_flow_pairs)

    if args.true_flow_pairs_manual_split:
        # Ensure both limits are specified for manual split
        if args.true_flow_pairs_training_amount is None or args.true_flow_pairs_testing_amount is None:
            raise ValueError("Both training and testing limits must be specified for manual split.")
        train_indexes, test_indexes = calc_train_test_index_manual_split(deepcorr_dataset, args.true_flow_pairs_training_amount, args.true_flow_pairs_testing_amount)
    else:
        # Implement automatic splitting logic
        train_indexes, test_indexes = calc_train_test_indexes_using_ratio(deepcorr_dataset, args.train_ratio)

    flow_pairs_train, labels_train, flow_pairs_test, labels_test = generate_flow_pairs_to_memmap(
        deepcorr_dataset, train_indexes, test_indexes, args.flow_size, args.save_directory, args.negative_samples)

    # Save args to a file in the save directory
    args_file_path = os.path.join(args.save_directory, 'script_args.txt')
    save_args_to_file(args, args_file_path)

    # Save shapes of the memmap arrays in a JSON file
    shapes = {
        "flow_pairs_train_shape": flow_pairs_train.shape,
        "labels_train_shape": labels_train.shape,
        "flow_pairs_test_shape": flow_pairs_test.shape,
        "labels_test_shape": labels_test.shape
    }
    shapes_file_path = os.path.join(args.save_directory, 'memmap_shapes.json')
    with open(shapes_file_path, 'w') as f:
        json.dump(shapes, f)

if __name__ == "__main__":
    main()
