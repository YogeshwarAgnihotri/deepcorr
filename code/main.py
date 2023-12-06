import os
import data_loader

# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

# parameters
negetive_samples=199
flow_size=300
training=True
num_epochs=200
negetive_samples=199
path_to_dataset='/home/yagnihotri/projects/deepcorr/dataset'
load_only_flows_with_min_300=True

dataset = data_loader.load_dataset(path_to_dataset=path_to_dataset, load_only_min_300_flows=load_only_flows_with_min_300)

print(len(dataset))