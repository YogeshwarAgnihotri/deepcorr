import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# evaluation.py
# Functions for evaluating the trained model and printing metrics.
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_predict
from shared.utils import save_plot_to_path, save_array_to_file
import matplotlib.pyplot as plt


def cross_validate(model, training_data, training_labels, roc_plot_enabled, run_folder_path, **kwargs):
    print("\nEvaluating the model on the training set with cross validation...")
    
    if roc_plot_enabled:
        # Ensure the model has the 'predict_proba' method
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model does not support probability predictions necessary for ROC curve analysis.")
        
        # Predict probabilities
        predicted_probabilities = cross_val_predict(model, training_data, training_labels, method='predict_proba', **kwargs)
        # Convert probabilities to binary predictions (1 if probability of positive class > 0.5)
        # done so we can use the standard metrics
        predicted_labels = (predicted_probabilities[:, 1] > 0.5).astype(int)
        print("Metrics for default threshold (0.5):")
        # Evaluate using standard metrics
        evaluate_model_print_metrics(true_labels=training_labels, predicted_labels=predicted_labels)

        save_array_to_file(predicted_probabilities, 'predicted_probabilities.npy')
        
        print("Roc curve log being created...")
        
        # Evaluate ROC curve with logarithmic scale
        fig, ax = generate_roc_plot(training_labels, predicted_probabilities[:, 1], log_scale=True)
        save_plot_to_path(fig, file_name='roc_curve_log_scale.png', save_path=run_folder_path)

        print("Roc curve being created...")

        # Evaluate ROC curve
        fig, ax = generate_roc_plot(training_labels, predicted_probabilities[:, 1], log_scale=False)
        save_plot_to_path(fig, file_name='roc_curve.png', save_path=run_folder_path)
    else:
        # Predict labels
        predicted_labels = cross_val_predict(model, training_data, training_labels, **kwargs)
        # Evaluate using standard metrics
        evaluate_model_print_metrics(true_labels=training_labels, predicted_labels=predicted_labels)


def evaluate_test_set(model, training_data, labels_test):
    print("\nEvaluating the model on the test set...")
    # # Make predictions and evaluate the model
    predictions = model.predict(training_data)
    evaluate_model_print_metrics(true_labels=labels_test, predicted_labels=predictions)
    
def evaluate_model_print_metrics(true_labels, predicted_labels):
    # Evaluate the model
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Print the evaluation results
    print(f'\nAccuracy: {accuracy:.2f}')
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

def generate_roc_plot(true_labels, predicted_scores, log_scale=False, point_markers=None):
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
    roc_auc = auc(fpr, tpr)

    save_array_to_file(fpr, 'fpr.npy')
    save_array_to_file(tpr, 'tpr.npy')
    save_array_to_file(thresholds, 'thresholds.npy')

    # Create ROC curve plot
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # Add scatter plot for points
    if point_markers is None:
        ax.scatter(fpr, tpr, color='darkorange', s=10)
    else:
        # Use custom markers if provided
        for i, marker in enumerate(point_markers):
            ax.scatter(fpr[i], tpr[i], color='darkorange', s=10, marker=marker)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")

    # Set x-axis to logarithmic scale if log_scale is True
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlim([1e-8, 1])  # Adjust limits as needed

    return fig, ax
