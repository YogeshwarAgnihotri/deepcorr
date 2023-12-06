import os
import data_loader
import train_test_split
import datetime
import pickle
from utlis import create_run_folder

############## PARAMETERS FOR BOTH ##############
gpu_device=0
training=True
path_dataset="/home/yagnihotri/projects/deepcorr/dataset"

############## PARAMETERS TRAINING ##############
load_only_flows_with_min_300=True

negative_samples=199
flow_size=300
num_epochs=200
#train_ratio=float(len_tr-6000)/float(len_tr)
train_ratio=0.8
path_for_saving_run="/home/yagnihotri/projects/deepcorr/runs"

############## PARAMETERS TESTING ##############
path_of_saved_run="/home/yagnihotri/projects/deepcorr/runs/Date_06-12-2023_19:47:15__NegativeSamples_199_FlowSize_300_NumEpochs_200_LoadOnlyFlowsWithMin300_True_TrainRatio_0.8"


############## CODE ##############
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

# TODO Change this up sometime. maybe with config files that show the full parameters
run_name = str("Date_{5}__NegativeSamples_{0}_FlowSize_{1}_NumEpochs_{2}_LoadOnlyFlowsWithMin300_{3}_TrainRatio_{4}".format(
    negative_samples, flow_size, num_epochs, load_only_flows_with_min_300, 
    train_ratio, datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")))

run_folder_path = create_run_folder(path_for_saving_run, run_name)

#dataset = data_loader.load_dataset(path_dataset=path_dataset, load_only_min_300_flows=load_only_flows_with_min_300)
#for testing code load only small dataset
dataset = pickle.load(open('/home/yagnihotri/projects/deepcorr/dataset/8802_tordata.pickle'))

if training:
    # split dataset into training and testing
    train_test_split.split_train_test(dataset, train_ratio, run_folder_path)    
    
else:
    # load testing indexes
    test_index = data_loader.load_test_index(path_of_saved_run)