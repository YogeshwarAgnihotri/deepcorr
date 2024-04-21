# Needed to find the other modules. Dont really like this solution.
import argparse
import csv
import sys
import os
import time

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay


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
from shared.utils import StreamToLogger, copy_file, create_path, create_run_folder, save_plot_to_path, setup_logger
from config_utlis import load_config

def evaluate_models(config, parent_folder_path):
    evaluation_results = []
    metrics_summary = []  # List to hold metrics summaries for each evaluation run

    for evaluation_run_name, eval_settings in config['evaluations'].items():
        print(f"\n---------------------------------- Evaluation run: {evaluation_run_name} -----------------------------------------")
        
        model_path = eval_settings['model_path']
        pregenerated_dataset_path = eval_settings['pregenerated_dataset_path']
        run_folder_path = os.path.join(parent_folder_path, evaluation_run_name)
        
        # Ensure the run folder exists
        os.makedirs(run_folder_path, exist_ok=True)

        model = joblib.load(model_path)

        # Assume load_pregenerated_memmap_dataset & other functions are defined elsewhere
        flow_pairs_train, labels_train, flow_pairs_test, labels_test = \
            load_pregenerated_memmap_dataset(pregenerated_dataset_path)
        
        flattened_flow_pairs_test = flatten_generated_flow_pairs(flow_pairs_test)[0]
        flattened_labels_test = flatten_arrays(labels_test)[0]

        # start timer
        start_time = time.time()
        fpr, tpr, roc_auc, thresholds = evaluate_test_set(model, flattened_flow_pairs_test, flattened_labels_test)
        end_time = time.time()
        print(f"Time taken to evaluate model: {end_time - start_time} seconds")

        # Check if evaluation_run_name matches the specified criteria
        if evaluation_run_name == "Decision Tree":
            # Filter out fpr and tpr values which are 0
            non_zero_indices = (fpr != 0) & (tpr != 0)
            fpr_filtered = fpr[non_zero_indices]
            tpr_filtered = tpr[non_zero_indices]

            evaluation_results.append((evaluation_run_name, (fpr_filtered, tpr_filtered, roc_auc, thresholds)))
        else:
            evaluation_results.append((evaluation_run_name, (fpr, tpr, roc_auc, thresholds)))

        print(evaluation_results)

        # Calculate additional metrics using binary predictions.
        y_pred_binary = [1 if score > 0.5 else 0 for score in model.predict_proba(flattened_flow_pairs_test)[:, 1]]
        accuracy = accuracy_score(flattened_labels_test, y_pred_binary)
        precision = precision_score(flattened_labels_test, y_pred_binary)
        recall = recall_score(flattened_labels_test, y_pred_binary)
        f1 = f1_score(flattened_labels_test, y_pred_binary)

        # Append metrics summary for the current evaluation run
        metrics_summary.append(f"{evaluation_run_name}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n")

        # Other evaluation steps...

    # Write the combined metrics summary to a file in the parent folder after the loop
    summary_file_path = os.path.join(parent_folder_path, "evaluation_metrics_summary.txt")
    with open(summary_file_path, 'w') as summary_file:
        for summary in metrics_summary:
            summary_file.write(summary)

    additional_roc_curves = config.get('additional_roc_curves', [])
    # ... (existing code) ...
    combined_roc_curves(evaluation_results, parent_folder_path, additional_roc_curves)



def combined_roc_curves(evaluation_results, destination_folder_path, additional_roc_curves):
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

    # Set the margins for the linear scale plots
    ax_linear.margins(0)
    ax_linear_no_scatter.margins(0)

    for ax in [ax_linear, ax_linear_no_scatter, ax_log, ax_log_no_scatter]:
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

    # Set log scale for log plots
    for ax in [ax_log, ax_log_no_scatter]:
        ax.set_xscale('log')
        ax.set_xlim(left=10**-5, right=1)  # Limit log scale to 10^0

    for eval_name, (fpr, tpr, roc_auc, thresholds) in evaluation_results:
        # Plot with scatter points
        line, = ax_linear.plot(fpr, tpr, label=f'{eval_name} (AUC = {roc_auc:.2f})')
        ax_linear.scatter(fpr, tpr, color=line.get_color(), s=10, zorder=2)
        fpr_log = np.clip(fpr, a_min=10**-5, a_max=None)
        line_log, = ax_log.plot(fpr_log, tpr, label=f'{eval_name} (AUC = {roc_auc:.2f})')
        ax_log.scatter(fpr_log, tpr, color=line_log.get_color(), s=10, zorder=2)

        # Plot without scatter points
        ax_linear_no_scatter.plot(fpr, tpr, label=f'{eval_name} (AUC = {roc_auc:.2f})')
        ax_log_no_scatter.plot(fpr_log, tpr, label=f'{eval_name} (AUC = {roc_auc:.2f})')

    # Now plot the additional ROC curves from the list if provided
    if additional_roc_curves:
        for roc_info in additional_roc_curves:
            try:
                with open(roc_info['path'], 'r') as csvfile:
                    csvreader = csv.reader(csvfile)
                    additional_fpr = []
                    additional_tpr = []
                    for row in csvreader:
                        additional_fpr.append(float(row[0]))
                        additional_tpr.append(float(row[1]))
                    
                    roc_label = roc_info['name']
                    ax_linear.plot(additional_fpr, additional_tpr, label=roc_label)
                    ax_linear_no_scatter.plot(additional_fpr, additional_tpr, label=roc_label)
                    additional_fpr_log = np.clip(additional_fpr, a_min=10**-5, a_max=None)
                    ax_log.plot(additional_fpr_log, additional_tpr, label=roc_label)
                    ax_log_no_scatter.plot(additional_fpr_log, additional_tpr, label=roc_label)
            except FileNotFoundError:
                print(f"Additional ROC CSV file not found at: {roc_info['path']}")

    for ax in [ax_linear, ax_log, ax_linear_no_scatter, ax_log_no_scatter]:
        ax.legend(loc='lower right', fontsize=5)

    for ax in [ax_log, ax_log_no_scatter]:
        ax.legend(loc='lower right', fontsize=5)

    # Save the combined ROC curves
    svg_path = os.path.join(destination_folder_path, "pdf")
    create_path(svg_path)
    save_plot_to_path(fig=fig_linear, file_name="combined_roc_curve_linear_with_scatter.pdf", save_path=svg_path)
    save_plot_to_path(fig=fig_linear_no_scatter, file_name="combined_roc_curve_linear.pdf", save_path=svg_path)
    save_plot_to_path(fig=fig_log, file_name="combined_roc_curve_log_with_scatter.pdf", save_path=svg_path)
    save_plot_to_path(fig=fig_log_no_scatter, file_name="combined_roc_curve_log.pdf", save_path=svg_path)

    save_plot_to_path(fig=fig_linear, file_name="combined_roc_curve_linear_with_scatter.png", save_path=destination_folder_path)
    save_plot_to_path(fig=fig_linear_no_scatter, file_name="combined_roc_curve_linear.png", save_path=destination_folder_path)
    save_plot_to_path(fig=fig_log, file_name="combined_roc_curve_log_with_scatter.png", save_path=destination_folder_path)
    save_plot_to_path(fig=fig_log_no_scatter, file_name="combined_roc_curve_log.png", save_path=destination_folder_path)


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

    copy_file(args.config_path, os.path.join(
        parent_run_folder_path, "used_config_train.yaml"))
    # Evaluate models and combine ROC curves
    evaluate_models(config, parent_run_folder_path)

if __name__ == "__main__":
    main()