import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, precision_recall_fscore_support, roc_curve,accuracy_score

from shared.utils import save_plot_to_path, save_array_to_file


def evaluate_test_set(model, training_data, labels_test, run_folder_path):
    print("\nEvaluating the model on the test set...")
    # Make predictions using predict_proba to get the probability scores
    scores = model.predict_proba(training_data)[:, 1]  # assuming the second column is for the positive class
    evaluate_model_print_metrics(true_labels=labels_test, predicted_labels=(scores > 0.5).astype(int))
    fig_linear, fig_roc = calc_roc_curves(true_labels=labels_test, 
                                          predicted_scores=scores,
                                          run_folder_path=run_folder_path)


    # Save the ROC curves to the run folder
    save_plot_to_path(fig_linear, "eval_roc_curve_linear.png", run_folder_path)
    save_plot_to_path(fig_roc, "eval_roc_curve_log.png", run_folder_path)

    
def evaluate_model_print_metrics(true_labels, predicted_labels):
    print("\nMetrics for threshold = 0.5")
    # Evaluate the model
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Print the evaluation results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    # Assuming conf_matrix is the result of confusion_matrix(labels_test, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()

    print("Confusion Matrix:")
    print(f"{'':>10} {'Predicted':>18}")
    print(f"{'':>10} {'0':>8} {'1':>8}")
    print(f"{'Actual 0':>10} {TN:>8} {FP:>8}")
    print(f"{'Actual 1':>10} {FN:>8} {TP:>8}")

    # Calculate True Positive Rate (TPR) also known as Recall
    TPR = TP / (TP + FN)  # TPR = Recall
    # Calculate False Positive Rate (FPR)
    FPR = FP / (FP + TN)
    print(f"True Positive Rate (TPR/Recall): {TPR:.2f}")
    print(f"False Positive Rate (FPR): {FPR:.2f}")

def calc_roc_curves(true_labels, predicted_scores, run_folder_path):
    """
    Plots ROC curves in both linear and logarithmic scales using RocCurveDisplay and returns the figure objects.
    It also plots the threshold points that make up the ROC curve.

    Args:
    true_labels (array-like): True binary labels.
    predicted_scores (array-like): Predicted scores or probabilities for the positive class.
    
    Returns:
    fig_linear, fig_log: Figures for the linear and logarithmic ROC curves with threshold points.
    """

    print("\nCalculating ROC curves on test set...")
    
    debug_run_folder_path = os.path.join(run_folder_path, "debug")

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
    roc_auc = auc(fpr, tpr)
    
    # Linear scale plot
    fig_linear, ax_linear = plt.subplots()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax_linear)
    # Plot the threshold points for the linear scale
    ax_linear.scatter(fpr, tpr, color='red', s=10, zorder=2)
    ax_linear.set_title('Receiver Operating Characteristic (Linear Scale)')

    # Logarithmic scale plot
    fig_log, ax_log = plt.subplots()
    # Set the minimum value for fpr to avoid log(0) and set lower bound for log scale to 10^-5
    fpr_log = np.clip(fpr, a_min=10**-5, a_max=None)
    RocCurveDisplay(fpr=fpr_log, tpr=tpr, roc_auc=roc_auc).plot(ax=ax_log)
    # Plot the threshold points for the logarithmic scale
    ax_log.scatter(fpr_log, tpr, color='red', s=10, zorder=2)
    ax_log.set_xscale('log')
    # Set limits for the x-axis to start from 10^-5
    ax_log.set_xlim(left=10**-5)
    ax_log.set_title('Receiver Operating Characteristic (Logarithmic Scale)')

    # Debug 
    save_array_to_file(array=np.column_stack((fpr, tpr, thresholds)), 
                       file_name="fpr_tpr_threshold", 
                       save_path=debug_run_folder_path)

    # Return the figure objects
    return fig_linear, fig_log