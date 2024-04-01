# Needed to find the other modules. Dont really like this solution.
import argparse
import sys
import os

from matplotlib import pyplot as plt
import numpy as np


sys.path.insert(0, 
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), '../../modules')))
sys.path.insert(0, 
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), '../../..')))

import joblib

from shared.model_evaluation import evaluate_test_set
from shared.data_handling import load_pregenerated_memmap_dataset
from shared.data_processing import flatten_generated_flow_pairs, flatten_arrays
from shared.utils import StreamToLogger, create_run_folder, save_plot_to_path, setup_logger
from config_utlis import load_config

def evaluate_models(config, parent_folder_path):
    evaluation_results = []
    for evaluation_run_name, eval_settings in config['evaluations'].items():
        print(f"\n---------------------------------- Evaluation run: {evaluation_run_name} -----------------------------------------")
        
        model_path = eval_settings['model_path']
        pregenerated_dataset_path = eval_settings['pregenerated_dataset_path']
        run_folder_path = os.path.join(parent_folder_path, evaluation_run_name)
        
        # Ensure the run folder exists
        os.makedirs(run_folder_path, exist_ok=True)

        model = joblib.load(model_path)

        flow_pairs_train, labels_train, flow_pairs_test, labels_test = \
            load_pregenerated_memmap_dataset(pregenerated_dataset_path)
        
        flattened_flow_pairs_test = flatten_generated_flow_pairs(flow_pairs_test)[0]
        flattened_labels_test = flatten_arrays(labels_test)[0]

        # Evaluate the model on the test set and collect results
        fpr, tpr, roc_auc, thresholds = evaluate_test_set(model, flattened_flow_pairs_test, flattened_labels_test)
        evaluation_results.append((evaluation_run_name, (fpr, tpr, roc_auc, thresholds)))
        
        # Output TPR, FPR, and threshold values to a text file
        metrics_file_path = os.path.join(run_folder_path, f"{evaluation_run_name}_metrics.txt")
        with open(metrics_file_path, 'w') as metrics_file:
            for fp, tp, th in zip(fpr, tpr, thresholds):
                metrics_file.write(f"FPR: {fp}, TPR: {tp}, Threshold: {th}\n")
    
    # Combine and plot ROC curves for all evaluations
    combined_roc_curves(evaluation_results, destination_folder_path=parent_folder_path)


def combined_roc_curves(evaluation_results, destination_folder_path):
    """
    Plots combined ROC curves in both linear and logarithmic scales for all evaluated models,
    creating four plots in total: two with scatter points (FPR, TPR) and two without.
    Args:
    evaluation_results (list): A list of tuples containing evaluation names and their corresponding metrics (fpr, tpr, roc_auc, thresholds).
    destination_folder_path (str): The path to the folder where the plots should be saved.
    """
    print("\nCalculating combined ROC curves...")

    # Prepare the plots
    fig_linear, ax_linear = plt.subplots()
    fig_linear_no_scatter, ax_linear_no_scatter = plt.subplots()
    fig_log, ax_log = plt.subplots()
    fig_log_no_scatter, ax_log_no_scatter = plt.subplots()

    for ax in [ax_linear, ax_linear_no_scatter, ax_log, ax_log_no_scatter]:
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

    # Set log scale for log plots
    for ax in [ax_log, ax_log_no_scatter]:
        ax.set_xscale('log')
        ax.set_xlim(left=10**-5, right=1)  # Limit log scale to 10^0

    for eval_name, (fpr, tpr, roc_auc, thresholds) in evaluation_results:
        # Plot with scatter points
        ax_linear.plot(fpr, tpr, label=f'{eval_name} (AUC = {roc_auc:.2f})')
        ax_linear.scatter(fpr, tpr, color='red', s=10, zorder=2)
        fpr_log = np.clip(fpr, a_min=10**-5, a_max=None)
        ax_log.plot(fpr_log, tpr, label=f'{eval_name} (AUC = {roc_auc:.2f})')
        ax_log.scatter(fpr_log, tpr, color='red', s=10, zorder=2)

        # Plot without scatter points
        ax_linear_no_scatter.plot(fpr, tpr, label=f'{eval_name} (AUC = {roc_auc:.2f})')
        ax_log_no_scatter.plot(fpr_log, tpr, label=f'{eval_name} (AUC = {roc_auc:.2f})')

    for ax in [ax_linear, ax_log, ax_linear_no_scatter, ax_log_no_scatter]:
        ax.legend(loc='lower right', fontsize=8)

    for ax in [ax_log, ax_log_no_scatter]:
        ax.legend(loc='upper left', fontsize=8)

    # Save the combined ROC curves
    save_plot_to_path(fig=fig_linear, file_name="combined_roc_curve_linear_with_scatter.svg", save_path=destination_folder_path)
    save_plot_to_path(fig=fig_linear_no_scatter, file_name="combined_roc_curve_linear.svg", save_path=destination_folder_path)
    save_plot_to_path(fig=fig_log, file_name="combined_roc_curve_log_with_scatter.svg", save_path=destination_folder_path)
    save_plot_to_path(fig=fig_log_no_scatter, file_name="combined_roc_curve_log.svg", save_path=destination_folder_path)


def main():
    """Main function to evaluate models based on a configuration file."""
    parser = argparse.ArgumentParser(description='Evaluate models based on a configuration file.')
    parser.add_argument('-c', '--config_path', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('-r', '--run_name', type=str, help='Name of the run followed by date time. If not set the current date and time only will be used.')
    args = parser.parse_args()

    config = load_config(args.config_path)

    parent_run_folder_path = create_run_folder(
        config['parent_folder_path'], args.run_name
    )
    output_file_path = os.path.join(parent_run_folder_path, "testing_log.txt")
    logger = setup_logger('TrainingLogger', output_file_path)
    sys.stdout = StreamToLogger(logger, sys.stdout)
    
    # Evaluate models and combine ROC curves
    evaluate_models(config, parent_run_folder_path)

if __name__ == "__main__":
    main()