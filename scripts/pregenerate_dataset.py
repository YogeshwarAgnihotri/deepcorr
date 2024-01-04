# quick cmd to run this script: 
""" 
python home/yagnihotri/projects/corr/scripts/pregenerate_dataset.py --dataset_path=/home/yagnihotri/datasets/deepcorr_original_dataset/ --train_ratio=0.8 --save_directory=/home/yagnihotri/datasets/deepcorr_custom_pregenerated --negative_samples=1 --load_all_data

"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json


from shared.data_loader import load_dataset_deepcorr
from shared.data_processing import generate_flow_pairs_to_memmap
from shared.train_test_split import calc_train_test_indexes

def save_args_to_file(args, file_path):
    with open(file_path, 'w') as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")

def main():
    parser = argparse.ArgumentParser(description='Split the deepcorr dataset into training and test indexes.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--save_directory', type=str, required=True, help='Directory to save the split indexes')
    parser.add_argument('--flow_size', type=int, default=300, help='Random seed')
    parser.add_argument('--negative_samples', type=int, default=1, help='Number of negative samples to include')
    parser.add_argument('--load_all_data', action='store_true', help='If true, load all data; otherwise, only load flows with minimum specified packets')

    args = parser.parse_args()

    deepcorr_dataset = load_dataset_deepcorr(args.dataset_path, args.load_all_data)
    train_indexes, test_indexes = calc_train_test_indexes(deepcorr_dataset, args.train_ratio)

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
