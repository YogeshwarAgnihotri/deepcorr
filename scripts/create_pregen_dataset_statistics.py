import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import argparse 
from shared.data_handling import load_dataset_deepcorr
import tqdm

#TODO STLL BUGGY, CANT REMEBER WHAT WAS WRONG WITH IT

def print_stats_for_flow_pairs(dataset, flow_size):
    # # Slicing the array for indices 0-3 and 4-7 in the second dimension
    # all_flow_pairs_ipd = flow_pairs[:, 0:4, :, :]
    # all_flow_pairs_packet_size = flow_pairs[:, 4:8, :, :]
    
    # # for debugging
    # save_array_to_file(all_flow_pairs_ipd, './all_flow_pairs_ipd.npy')
    # save_array_to_file(all_flow_pairs_packet_size, './all_flow_pairs_packet_size.npy')

    # # Calculating overall average for indices 0-3
    # avg_ipd = np.mean(all_flow_pairs_ipd)
    # min_ipd = np.min(all_flow_pairs_ipd)
    # max_ipd = np.max(all_flow_pairs_ipd)
    # std_ipd = np.std(all_flow_pairs_ipd)

    # # Calculating overall average for indices 4-7
    # avg_packet_size = np.mean(all_flow_pairs_packet_size)
    # min_packet_size = np.min(all_flow_pairs_packet_size)
    # max_packet_size = np.max(all_flow_pairs_packet_size)
    # std_packet_size = np.std(all_flow_pairs_packet_size)

    # print("Average IPD: {}".format(avg_ipd))
    # print("Min IPD: {}".format(min_ipd))
    # print("Max IPD: {}".format(max_ipd))
    # print("Std IPD: {}".format(std_ipd))
    # print("Average packet size: {}".format(avg_packet_size))
    # print("Min packet size: {}".format(min_packet_size))
    # print("Max packet size: {}".format(max_packet_size))
    # print("Std packet size: {}".format(std_packet_size))
    l2s = np.zeros((len(dataset),8,flow_size,1))
    dataset_size = len(dataset)
    
    for i in tqdm.tqdm(dataset_size, desc="Loading dataset"):
        l2s[i,0,:,0]=np.array(dataset[i]['here'][0]['<-'][:flow_size])*1000.0
        l2s[i,1,:,0]=np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0
        l2s[i,2,:,0]=np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0
        l2s[i,3,:,0]=np.array(dataset[i]['here'][0]['->'][:flow_size])*1000.0

        # "There/here"[1] are the packet sizes
        l2s[i,4,:,0]=np.array(dataset[i]['here'][1]['<-'][:flow_size])/1000.0
        l2s[i,5,:,0]=np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0
        l2s[i,6,:,0]=np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0
        l2s[i,7,:,0]=np.array(dataset[i]['here'][1]['->'][:flow_size])/1000.0



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Print dataset statistics of deepcorr dataset')
    parser.add_argument('--dataset_path', default= "/home/yagnihotri/datasets/deepcorr_original_dataset/" ,type=str, required=True, help='Path to the base dataset')

    # Parse arguments
    args = parser.parse_args()

    dataset = load_dataset_deepcorr(args.dataset_path, load_all_data=False)

    # Calculate statistics for training set
    print(f"Calculating statistics for dataset at {args.dataset_path}...")
    print(np.shape(dataset))

if __name__ == "__main__":
    main()