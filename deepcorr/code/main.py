import os
import datetime
import pickle
import tensorflow as tf

import data_loader
import train_test_split
from utils import create_run_folder
from model import build_graph_testing, build_graph_training
from train_test import train_model,test_model

############## PARAMETERS FOR BOTH ##############
gpu_device = 0
training = True
path_dataset = "/home/yagnihotri/datasets/deepcorr_original_dataset"

############## PARAMETERS TRAINING ##############
load_only_flows_with_min_300 = True

negative_samples = 199
flow_size = 300
num_epochs = 200
# train_ratio = float(len_tr-6000)/float(len_tr)
train_ratio = 0.7
batch_size_training = 256
learn_rate = 0.0001

# paths
path_for_saving_run = "/home/yagnihotri/projects/deepcorr/runs"

############## PARAMETERS TESTING ##############
batch_size_testing = 256
path_of_saved_run = "/home/yagnihotri/projects/deepcorr/runs/Date_06-12-2023_19:47:15__NegativeSamples_199_FlowSize_300_NumEpochs_200_LoadOnlyFlowsWithMin300_True_TrainRatio_0.8"

############## CODE ##############
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

# TODO Change this up sometime. maybe with config files that show the full 
# parameters
run_name = f"Date_{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}__NegativeSamples_{negative_samples}_FlowSize_{flow_size}_NumEpochs_{num_epochs}_LoadOnlyFlowsWithMin300_{load_only_flows_with_min_300}_TrainRatio_{train_ratio}"

run_folder_path = create_run_folder(path_for_saving_run, run_name)

dataset = data_loader.load_dataset(path_dataset=path_dataset, load_only_min_300_flows = load_only_flows_with_min_300)
#for testing code load only small dataset
#dataset = pickle.load(open('/home/yagnihotri/projects/deepcorr/dataset/8802_tordata300.pickle', 'rb'))

if training:
    # split dataset into training and testing
    train_indexes, test_indexes = train_test_split.split_train_test(dataset, train_ratio, run_folder_path)
    
    train_test_split.save_test_indexes_to_path(test_indexes, run_folder_path)

    train_flow_before, train_label, dropout_keep_prob, loss, optimizer, summary_op, init, saver, predict, graph = build_graph_training(batch_size_training, flow_size, learn_rate)

    train_model(num_epochs, dataset, train_indexes, test_indexes, flow_size, negative_samples, batch_size_training, train_flow_before, train_label, dropout_keep_prob, loss, optimizer, summary_op, init, saver, predict, graph, run_folder_path)
else:
    # load testing indexes
    test_index = data_loader.load_test_index(path_of_saved_run)

    train_flow_before, train_label, dropout_keep_prob, saver, predict, graph = build_graph_testing(batch_size_testing, flow_size)