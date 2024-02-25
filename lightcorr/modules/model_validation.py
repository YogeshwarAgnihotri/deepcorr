import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve
from sklearn.model_selection import cross_validate, KFold

from shared.utils import save_plot_to_path, save_array_to_file, create_path
from modules.plotting import generate_custom_cv_plots

import numpy as np
from tqdm import tqdm

def perform_custom_cv(trained_model, 
               flattened_flow_pairs_train, 
               flattened_labels_train, 
               cv_num,
               run_folder_path):
    
    cv_res, mean_res, std_res = custom_cv(trained_model, 
                                  cv_num, 
                                  flattened_flow_pairs_train, 
                                  flattened_labels_train, 
                                  run_folder_path, 
                                  start_with_non_zero_fpr=False)
    
    fig_linear, fig_linear_threshold_points, fig_linear_no_mean, \
    fig_linear_no_mean_threshold_points, fig_log, fig_log_threshold_points, \
    fig_log_no_mean, fig_log_no_mean_threshold_points, = \
    generate_custom_cv_plots(cv_res, mean_res, std_res)

    # Save the plots
    linear_folder = os.path.join(run_folder_path, "linear")
    create_path(linear_folder)
    save_plot_to_path(fig=fig_linear, file_name="roc_linear.png", save_path=linear_folder)
    save_plot_to_path(fig=fig_linear_threshold_points, file_name="roc_linear_threshold_points.png", save_path=linear_folder)
    save_plot_to_path(fig=fig_linear_no_mean, file_name="roc_linear_no_mean.png", save_path=linear_folder)
    save_plot_to_path(fig=fig_linear_no_mean_threshold_points, file_name="roc_linear_no_mean_threshold_points.png", save_path=linear_folder)

    log_folder = os.path.join(run_folder_path, "log")
    create_path(log_folder)
    save_plot_to_path(fig=fig_log, file_name="roc_log.png", save_path=log_folder)
    save_plot_to_path(fig=fig_log_threshold_points, file_name="roc_log_threshold_points.png", save_path=log_folder)
    save_plot_to_path(fig=fig_log_no_mean, file_name="roc_log_no_mean.png", save_path=log_folder)
    save_plot_to_path(fig=fig_log_no_mean_threshold_points, file_name="roc_log_no_mean_threshold_points.png", save_path=log_folder)

    print(f"\nMean Accuracy: {mean_res['mean_accuracy']:.2f}")
    print(f"Mean Precision: {mean_res['mean_precision']:.2f}")
    print(f"Mean Recall: {mean_res['mean_recall']:.2f}")
    print(f"Mean F1 Score: {mean_res['mean_f1']:.2f}")
    
    return mean_res
  
def custom_cv(model, cv, X, y, run_folder_path, start_with_non_zero_fpr=False):
    n_splits = cv
    cv = KFold(n_splits=n_splits)
    interpolation_points = 100

    # Preparing the results dictionary to store all necessary data
    cv_res = {
        'fpr_linear': [],
        'tpr_linear': [],
        'thresholds_linear': [],
        'aucs_linear': [],
        'fpr_log': [],
        'tpr_log': [],
        'thresholds_log': [],
        'aucs_log': [],
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': [],
    }

    mean_res = {
        'mean_accuracy': [],
        'mean_precision': [],
        'mean_recall': [],
        'mean_f1': [],
        'mean_fpr_linear': np.linspace(0, 1, interpolation_points),
        'mean_tpr_linear' : [],
        'mean_fpr_log': np.logspace(-10, 0, interpolation_points),  # Placeholder, adjust based on min FPR
        'mean_tpr_log' : [],
        'mean_auc_linear': [],
        'mean_auc_log': [],
    }

    std_res = {
        'std_auc_linear': [],
        'std_auc_log': [],
    }

    min_non_zero_log_fpr = 1  # For adjusting log scale FPR

    # Initialize tqdm around the cross-validation loop
    with tqdm(total=cv.get_n_splits(), desc="Cross-validating") as pbar:
        for fold, (train, test) in enumerate(cv.split(X, y)):
            model.fit(X[train], y[train])

            # Get the score of the positive class
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X[test])
            else:
                scores = model.predict_proba(X[test])[:, 1]

            y_pred = model.predict(X[test])

            fpr_linear, tpr_linear, thresholds_linear = roc_curve(y[test], scores)
            auc_linear = auc(fpr_linear, tpr_linear)

            if start_with_non_zero_fpr:
                # Filter out the zero values from fpr for log scale plotting
                non_zero_indices = fpr_linear > 0
                fpr_log = fpr_linear[non_zero_indices]
                tpr_log = tpr_linear[non_zero_indices]
                thresholds_log = thresholds_linear[non_zero_indices]
                # Update the minimum non-zero FPR value
                if len(fpr_log) > 0:
                    fold_min_non_zero_fpr = fpr_log.min()
                    # Redefine mean_fpr_log based on the smallest non-zero FPR found
                    min_non_zero_log_fpr = min(min_non_zero_log_fpr, fold_min_non_zero_fpr)
            else:
                fpr_log = np.clip(fpr_linear, a_min=10**-5, a_max=None)
                tpr_log = tpr_linear
                thresholds_log = thresholds_linear

            auc_log = auc(fpr_log, tpr_log)

            # Append metrics and ROC curve data to the results dictionary
            cv_res['fpr_linear'].append(fpr_linear)
            cv_res['tpr_linear'].append(tpr_linear)
            cv_res['thresholds_linear'].append(thresholds_linear)
            cv_res['aucs_linear'].append(auc_linear)

            cv_res['fpr_log'].append(fpr_log)
            cv_res['tpr_log'].append(tpr_log)
            cv_res['thresholds_log'].append(thresholds_log)
            cv_res['aucs_log'].append(auc_log)

            cv_res['accuracies'].append(accuracy_score(y[test], y_pred))
            cv_res['precisions'].append(precision_score(y[test], y_pred, zero_division=0))
            cv_res['recalls'].append(recall_score(y[test], y_pred))
            cv_res['f1_scores'].append(f1_score(y[test], y_pred))

            pbar.update(1)

    # After the loop, adjust the mean_fpr_log based on the smallest non-zero FPR found
    if start_with_non_zero_fpr and min_non_zero_log_fpr < 1:
        mean_res['mean_fpr_log'] = np.logspace(np.log10(min_non_zero_log_fpr), 0, interpolation_points)

    # Calculate mean and standard deviation for TPR
    mean_res['mean_tpr_linear'] = np.mean([np.interp(mean_res['mean_fpr_linear'], fpr, tpr) for fpr, tpr in zip(cv_res['fpr_linear'], cv_res['tpr_linear'])], axis=0)
    mean_res['mean_tpr_linear'][-1] = 1.0
    mean_res['mean_auc_linear'] = auc(mean_res['mean_fpr_linear'], mean_res['mean_tpr_linear'])
    std_res['std_auc_linear'] = np.std(cv_res['aucs_linear'])
    
    mean_res['mean_tpr_log'] = np.mean([np.interp(mean_res['mean_fpr_log'], fpr, tpr) for fpr, tpr in zip(cv_res['fpr_log'], cv_res['tpr_log'])], axis=0)
    mean_res['mean_tpr_log'][-1] = 1.0
    mean_res['mean_auc_log'] = auc(mean_res['mean_fpr_log'], mean_res['mean_tpr_log'])
    std_res['std_auc_log'] = np.std(cv_res['aucs_log'])

    # Compute and return summary statistics
    mean_res['mean_accuracy'] = np.mean(cv_res['accuracies'])
    mean_res['mean_precision'] = np.mean(cv_res['precisions'])
    mean_res['mean_recall'] = np.mean(cv_res['recalls'])
    mean_res['mean_f1'] = np.mean(cv_res['f1_scores'])

    return cv_res, mean_res, std_res