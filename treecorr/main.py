import argparse
import datetime
import logging
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


from shared.utils import create_path
from shared.data_loader import load_dataset_deepcorr
from shared.data_processing import generate_flow_pairs 
from shared.train_test_split import calc_train_test_indexes

import time

#################### Parameters ####################
# Set up argument parser
parser = argparse.ArgumentParser(description='Train a Decision Tree Classifier on DeepCorr dataset.')
parser.add_argument('--dataset_path', type=str, default="/home/yagnihotri/datasets/deepcorr_original_dataset", help='Path to the DeepCorr dataset')
parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
parser.add_argument('--flow_size', type=int, default=300, help='Flow size')
parser.add_argument('--negative_samples', type=int, default=199, help='Number of negative samples')
parser.add_argument('--load_all_data', action='store_true', help='If NOT set, only flows with minimum 300 packets will be loaded (about 7300 flow pairs). If set, all will be loaded (about 20-25k).')

# Parse arguments
args = parser.parse_args()

# Extracting arguments
dataset_path = args.dataset_path
train_ratio = args.train_ratio
flow_size = args.flow_size
negative_samples = args.negative_samples
load_all_data = args.load_all_data

class StreamToLogger:
    """
    Custom stream object that redirects writes to both a logger and the original stdout.
    """
    def __init__(self, logger, orig_stdout):
        self.logger = logger
        self.orig_stdout = orig_stdout
        self.linebuf = ''  # Buffer to accumulate lines until a newline character

    def write(self, message):
        """
        Write the message to logger and the original stdout.
        Only log messages when a newline character is encountered.
        """
        self.linebuf += message
        if '\n' in message:
            self.flush()

    def flush(self):
        """Flush the stream by logging the accumulated line and clearing the buffer."""
        if self.linebuf.rstrip():
            self.logger.info(self.linebuf.rstrip())
        self.orig_stdout.write(self.linebuf)
        self.linebuf = ''

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger."""
    formatter = logging.Formatter('%(message)s')  # Only include the message in logs

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    return logger


#################### Paths ####################
path_for_saving_run = "/home/yagnihotri/projects/corr/treecorr/runs"
# TODO Change run_name to something shorter up sometime. maybe with config files that show the full parameters
run_name = f"Date_{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
run_folder_path = os.path.join(path_for_saving_run, run_name)
create_path(run_folder_path)

output_file_path = os.path.join(run_folder_path, "training_log.txt")
logger = setup_logger('DeepCorrTraining', output_file_path)

# Redirect stdout
sys.stdout = StreamToLogger(logger, sys.stdout)

# Loading deepcorr dataset
deepcorr_dataset = load_dataset_deepcorr(dataset_path, load_all_data)

# Split the dataset into training and test sets
train_indexes, test_indexes = calc_train_test_indexes(deepcorr_dataset, train_ratio)

# Preprocess the data and generate the data arrays for training and testing
l2s, labels,l2s_test,labels_test = generate_flow_pairs(deepcorr_dataset, train_indexes, test_indexes, flow_size, run_folder_path, negative_samples)

l2s_flattened = l2s.reshape(l2s.shape[0], -1)

l2s_test_flattened = l2s_test.reshape(l2s_test.shape[0], -1)  # Same for test data

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()

# Start timing the training process
start_time = time.time()

print("\nTraining the model...")
# Train the model
clf.fit(l2s_flattened, labels)

# End timing the training process
training_time = time.time() - start_time
print(f'Training completed in {training_time:.2f} seconds')

# Start timing the prediction process
start_time = time.time()

# Make predictions
y_pred = clf.predict(l2s_test_flattened)

# End timing the prediction process
testing_time = time.time() - start_time
print(f'Testing completed in {testing_time:.2f} seconds')

# Evaluate the model
accuracy = accuracy_score(labels_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(labels_test, y_pred, average='binary')
conf_matrix = confusion_matrix(labels_test, y_pred)

# Print the evaluation results
print(f'\nAccuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

print('Confusion Matrix:')
# Define column and row labels
col_labels = ['Predicted Positive (0)', 'Predicted Negative (1)']
row_labels = ['Actual Positive (0)', 'Actual Negative (1)']
# Print column labels
print(f"{'':>20} {' '.join(f'{label:>20}' for label in col_labels)}")
# Print each row with its label
for label, row in zip(row_labels, conf_matrix):
    print(f'{label:>20} {" ".join(f"{num:20d}" for num in row)}')

# True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
TP = conf_matrix[1, 1]
FP = conf_matrix[0, 1]
TN = conf_matrix[0, 0]
FN = conf_matrix[1, 0]
# Calculate True Positive Rate (TPR) also known as Recall
TPR = TP / (TP + FN)  # TPR = Recall
# Calculate False Positive Rate (FPR)
FPR = FP / (FP + TN)
print(f"True Positive Rate (TPR/Recall): {TPR:.2f}")
print(f"False Positive Rate (FPR): {FPR:.2f}")


# print parameters which were used for this run
print("\nParameters:")
print(f"dataset_path: {dataset_path}")
print(f"train_ratio: {train_ratio}")
print(f"flow_size: {flow_size}")
print(f"negative_samples: {negative_samples}")
print(f"load_all_data: {load_all_data}")