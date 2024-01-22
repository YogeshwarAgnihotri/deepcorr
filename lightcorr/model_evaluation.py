import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import cross_validate
from shared.utils import save_plot_to_path, save_array_to_file
import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import KFold

import numpy as np


def custom_cross_validate(model, training_data, training_labels, roc_plot_enabled, run_folder_path, **kwargs):
    print("\nEvaluating the model on the training set with cross validation...")
    
    if roc_plot_enabled:
        fig, ax, fig_log, ax_log, mean_accuracy, mean_precision, mean_recall, mean_f1 \
        = custom_cv_roc(model, kwargs.get('cv'), training_data, training_labels)
        save_plot_to_path(fig, file_name='roc_curve.png', save_path=run_folder_path)
        save_plot_to_path(fig_log, file_name='roc_curve_log.png', save_path=run_folder_path)
        print(f"\nMean Accuracy: {mean_accuracy:.2f}")
        print(f"Mean Precision: {mean_precision:.2f}")
        print(f"Mean Recall: {mean_recall:.2f}")
        print(f"Mean F1 Score: {mean_f1:.2f}")
    else:
        # Just print the metrics specified in the config file
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        results = cross_validate(model, training_data, training_labels, scoring=scoring, **kwargs)
        print(results)

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

def custom_cv_roc(model, cv, X, y):
    n_splits = cv
    cv = KFold(n_splits=n_splits)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Lists to store metrics for each fold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Create two figures: one for linear and one for log scale
    fig_linear, ax_linear = plt.subplots(figsize=(6, 6))
    fig_log, ax_log = plt.subplots(figsize=(6, 6))

    for fold, (train, test) in enumerate(cv.split(X, y)):
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])

        # Calculate metrics
        accuracies.append(accuracy_score(y[test], y_pred))
        precisions.append(precision_score(y[test], y_pred, zero_division=0))
        recalls.append(recall_score(y[test], y_pred))
        f1_scores.append(f1_score(y[test], y_pred))

        # Plot for both linear and log scale
        for ax in [ax_linear, ax_log]:
            viz = RocCurveDisplay.from_estimator(
                model,
                X[test],
                y[test],
                name=f"ROC fold {fold}",
                alpha=0.3,
                lw=1,
                ax=ax
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Linear scale plot
    ax_linear.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    ax_linear.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve with variability (Linear Scale)",
    )
    ax_linear.legend(loc="lower right")

    # Log scale plot
    ax_log.set_xscale('log')
    ax_log.set_xlim(10**-5, 1)
    ax_log.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    ax_log.set(
        xlabel="False Positive Rate (Log Scale)",
        ylabel="True Positive Rate",
        title="Mean ROC curve with variability (Log Scale)",
    )
    ax_log.legend(loc="lower right")

    # Averages of the metrics
    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)

    return fig_linear, ax_linear, fig_log, ax_log, mean_accuracy, mean_precision, mean_recall, mean_f1
