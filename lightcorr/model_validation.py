import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve
from sklearn.model_selection import cross_validate, KFold
from shared.utils import save_plot_to_path, save_array_to_file, create_path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def custom_cross_validate(model, training_data, training_labels, roc_plot_enabled, run_folder_path, **kwargs):
    print("\nEvaluating the model on the training set with cross validation...")
    
    if roc_plot_enabled:
        fig_linear, ax_linear, fig_log, ax_log, fig_linear_no_mean, ax_linear_no_mean, \
        fig_log_no_mean, ax_log_no_mean, mean_accuracy, mean_precision, mean_recall, mean_f1 \
        = custom_cv_roc(model, kwargs.get('cv'), training_data, training_labels, run_folder_path)
        save_plot_to_path(fig_linear, file_name='roc_curve_mean.png', save_path=run_folder_path)
        save_plot_to_path(fig_log, file_name='roc_curve_log_mean.png', save_path=run_folder_path)
        save_plot_to_path(fig_linear_no_mean, file_name='roc_curve_no_mean.png', save_path=run_folder_path)
        save_plot_to_path(fig_log_no_mean, file_name='roc_curve_log_no_mean.png', save_path=run_folder_path)
        print(f"\nMean Accuracy: {mean_accuracy:.2f}")
        print(f"Mean Precision: {mean_precision:.2f}")
        print(f"Mean Recall: {mean_recall:.2f}")
        print(f"Mean F1 Score: {mean_f1:.2f}")
    else:
        # Just print the metrics specified in the config file
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        results = cross_validate(model, training_data, training_labels, scoring=scoring, **kwargs)
        print(results)

def custom_cv_roc(model, cv, X, y, run_folder_path, plot_threshhold_points=False):
    debug_run_folder_path = os.path.join(run_folder_path, "debug")
    create_path(debug_run_folder_path)
    n_splits = cv
    cv = KFold(n_splits=n_splits)
    interpolation_points = 100

    mean_fpr_linear = np.linspace(0, 1, interpolation_points)
    #the -10 this gets changed later to the minimum non-zero value of fpr_log
    mean_fpr_log = np.logspace(-10, 0, interpolation_points)

    tprs_linear = []
    aucs_linear = []

    tprs_log = []
    aucs_log = []

    min_non_zero_log_fpr = 1
    
    save_array_to_file(mean_fpr_linear, os.path.join(debug_run_folder_path, f"mean_fpr_linear.txt"))

    # Lists to store metrics for each fold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Create two figures: one for linear and one for log scale
    fig_linear, ax_linear = plt.subplots(figsize=(6, 6))
    fig_log, ax_log = plt.subplots(figsize=(6, 6))

    # Additional figures without the mean curve
    fig_linear_no_mean, ax_linear_no_mean = plt.subplots(figsize=(6, 6))
    fig_log_no_mean, ax_log_no_mean = plt.subplots(figsize=(6, 6))

    # Track the start time of cross-validation
    start_time = time.time()

    # Initialize tqdm around the cross-validation loop
    with tqdm(total=cv.get_n_splits(), desc="Cross-validating") as pbar:
        for fold, (train, test) in enumerate(cv.split(X, y)):
            fold_start = time.time()  # Start time for the current fold
            
            model.fit(X[train], y[train])
            
            # Get the score of the positive class
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X[test])
                print("decision_function")
            else:
                scores = model.predict_proba(X[test])[:, 1]
            
            y_pred = model.predict(X[test])

            save_array_to_file(scores, os.path.join(debug_run_folder_path, f"fold_{fold}_scores.txt"))
            
            # Calculate metrics
            accuracies.append(accuracy_score(y[test], y_pred))
            precisions.append(precision_score(y[test], y_pred, zero_division=0))
            recalls.append(recall_score(y[test], y_pred))
            f1_scores.append(f1_score(y[test], y_pred))
            
            # Compute ROC curve with many thresholds
            fpr_linear, tpr_linear, thresholds_linear = roc_curve(y[test], scores, drop_intermediate=False)

            # Filter out the zero values from fpr for log scale plotting
            # This is because log(0) is undefined, and the curves should start at the 
            # first non-zero value of fpr
            non_zero_indices = fpr_linear > 0
            fpr_log = fpr_linear[non_zero_indices]
            tpr_log = tpr_linear[non_zero_indices]
            thresholds_log = thresholds_linear[non_zero_indices]

            # Update the minimum non-zero FPR value
            if len(fpr_log) > 0:
                fold_min_non_zero_fpr = fpr_log.min()
                min_non_zero_log_fpr = min(min_non_zero_log_fpr, fold_min_non_zero_fpr)
                # Redefine mean_fpr_log based on the smallest non-zero FPR found
                mean_fpr_log = np.logspace(np.log10(min_non_zero_log_fpr), 0, interpolation_points)
                #print(f"mean_fpr_log: {mean_fpr_log}")
                #mean_fpr_log = mean_fpr_log[1:]  # Remove the first element of mean_fpr_log because it is 0
                #print(f"mean_fpr_log: {mean_fpr_log}")


            # Combine the arrays into a single array of triplets for debugging
            save_array_to_file(np.column_stack((fpr_linear, tpr_linear, thresholds_linear)), 
                               os.path.join(debug_run_folder_path, f"fold_{fold}_fpr_tpr_threshold_linear.txt"))
            save_array_to_file(np.column_stack((fpr_log, tpr_log, thresholds_log)), 
                               os.path.join(debug_run_folder_path, f"fold_{fold}_fpr_tpr_threshold_log.txt"))
            
            ax_linear.plot(fpr_linear, tpr_linear, lw=1, alpha=0.3, label=f'Fold {fold}')

            interp_tpr_linear = np.interp(mean_fpr_linear, fpr_linear, tpr_linear)
            save_array_to_file(interp_tpr_linear, os.path.join(debug_run_folder_path, f"fold_{fold}_interp_tpr_linear.txt"))
            interp_tpr_linear[0] = 0.0
            tprs_linear.append(interp_tpr_linear)
            aucs_linear.append(auc(fpr_linear, tpr_linear))

            ax_log.plot(fpr_log, tpr_log, lw=1, alpha=0.3, label=f'Fold {fold}')

            ax_linear_no_mean.plot(fpr_linear, tpr_linear, lw=1, alpha=0.3, label=f'Fold {fold}')
            ax_log_no_mean.plot(fpr_log, tpr_log, lw=1, alpha=0.3, label=f'Fold {fold}')
            
            if plot_threshhold_points:
                ax_linear.scatter(fpr_linear, tpr_linear, s=10)
                ax_log.scatter(fpr_log, tpr_log, s=10)  # Add ROC points

            interp_tpr_log = np.interp(mean_fpr_log, fpr_log, tpr_log)
            save_array_to_file(interp_tpr_log, os.path.join(debug_run_folder_path, f"fold_{fold}_interp_tpr_log.txt"))
            interp_tpr_log[0] = 0.0
            tprs_log.append(interp_tpr_log)
            aucs_log.append(auc(fpr_log, tpr_log))
            
            fold_end = time.time()  # End time for the current fold
            pbar.write(f"Finished fold {fold+1}/{n_splits}. Time taken: {fold_end - fold_start:.2f} seconds.")
            pbar.update(1)
            
    total_duration = time.time() - start_time
    tqdm.write(f"All folds completed. Total time taken: {total_duration:.2f} seconds.")


    save_array_to_file(mean_fpr_log, os.path.join(debug_run_folder_path, f"mean_fpr_log.txt"))
    
    save_array_to_file(np.array(tprs_linear), os.path.join(debug_run_folder_path, f"tprs_linear.txt"))
    mean_tpr_linear = np.mean(tprs_linear, axis=0)
    save_array_to_file(mean_tpr_linear, os.path.join(debug_run_folder_path, f"mean_tpr.txt"))
    mean_tpr_linear[-1] = 1.0
    mean_auc_linear = auc(mean_fpr_linear, mean_tpr_linear)
    std_auc_linear = np.std(aucs_linear)

    save_array_to_file(np.array(tprs_log), os.path.join(debug_run_folder_path, f"tprs_log.txt"))
    mean_tpr_log = np.mean(tprs_log, axis=0)
    save_array_to_file(mean_tpr_log, os.path.join(debug_run_folder_path, f"mean_tpr_log.txt"))
    mean_tpr_log[-1] = 1.0
    mean_auc_log = auc(mean_fpr_log, mean_tpr_log)
    std_auc_log = np.std(aucs_log)

    # Combine the arrays into a single array of triplets
    combined_array = np.column_stack((mean_fpr_linear, mean_tpr_linear))
    save_array_to_file(combined_array, os.path.join(debug_run_folder_path, f"mean_fpr_tpr_linear.txt"))

    combined_array = np.column_stack((mean_fpr_log, mean_tpr_log))
    save_array_to_file(combined_array, os.path.join(debug_run_folder_path, f"mean_fpr_tpr_log.txt"))

    # Linear scale plot
    ax_linear.plot(
        mean_fpr_linear, 
        mean_tpr_linear,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_linear, std_auc_linear),
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
        mean_fpr_log,  # This should also be offset if mean_fpr contains 0
        mean_tpr_log,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_log, std_auc_log),
        lw=2,
        alpha=0.8,
    )
    ax_log.set(
        xlabel="False Positive Rate (Log Scale)",
        ylabel="True Positive Rate",
        title="Mean ROC curve with variability (Log Scale)",
    )

    # Finalize the figures without the mean curve
    ax_linear_no_mean.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Individual ROC curves (Linear Scale)"
    )
    ax_linear_no_mean.legend(loc="lower right")

    ax_log_no_mean.set_xscale('log')
    ax_log_no_mean.set_xlim(10**-5, 1)
    ax_log_no_mean.set(
        xlabel="False Positive Rate (Log Scale)",
        ylabel="True Positive Rate",
        title="Individual ROC curves (Log Scale)"
    )
    ax_log_no_mean.legend(loc="upper left", fontsize='small', ncol=1, frameon=False)

    if plot_threshhold_points:
        ax_linear.scatter(mean_fpr_linear, mean_tpr_linear, s=10, color="b")  # Add ROC points
        ax_log.scatter(mean_fpr_log, mean_tpr_log, s=10, color="b")  # Add ROC points

    # For the log scale plot legend
    ax_log.legend(loc="upper left", fontsize='small', ncol=1, frameon=False)


    # Averages of the metrics
    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)

    return fig_linear, ax_linear, fig_log, ax_log, fig_linear_no_mean, ax_linear_no_mean, fig_log_no_mean, ax_log_no_mean, mean_accuracy, mean_precision, mean_recall, mean_f1

