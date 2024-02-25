import matplotlib.pyplot as plt

from sklearn.metrics import auc
# Todo this function can be made modular and more readable
def generate_custom_cv_plots(cv_res, mean_res, std_res):
    # Extract variables from the results dictionary
    fpr_linear_list = cv_res['fpr_linear']
    tpr_linear_list = cv_res['tpr_linear']   
    fpr_log_list = cv_res['fpr_log']
    tpr_log_list = cv_res['tpr_log']

    mean_fpr_linear = mean_res['mean_fpr_linear']
    mean_tpr_linear = mean_res['mean_tpr_linear']
    
    mean_fpr_log = mean_res['mean_fpr_log']
    mean_tpr_log = mean_res['mean_tpr_log']

    mean_auc_linear = mean_res['mean_auc_linear']
    std_auc_linear = std_res['std_auc_linear']

    mean_auc_log = mean_res['mean_auc_log']
    std_auc_log = std_res['std_auc_log']
    

    # Prepare plot figures
    fig_linear, ax_linear = plt.subplots(figsize=(6, 6))
    fig_linear_threshold_points, ax_linear_threshold_points = plt.subplots(figsize=(6, 6))
    fig_linear_no_mean, ax_linear_no_mean = plt.subplots(figsize=(6, 6))
    fig_linear_no_mean_threshold_points, ax_linear_no_mean_threshold_points = plt.subplots(figsize=(6, 6))
    
    fig_log, ax_log = plt.subplots(figsize=(6, 6))
    fig_log_threshold_points, ax_log_threshold_points = plt.subplots(figsize=(6, 6))
    fig_log_no_mean, ax_log_no_mean = plt.subplots(figsize=(6, 6))
    fig_log_no_mean_threshold_points, ax_log_no_mean_threshold_points = plt.subplots(figsize=(6, 6))
    
    for fold, (fpr_linear, tpr_linear, fpr_log, tpr_log) in enumerate(zip(fpr_linear_list, tpr_linear_list, fpr_log_list, tpr_log_list)):
        fold_label = f'Fold {fold + 1}'  # Fold numbering starts at 1 for readability
        
        ax_linear.plot(fpr_linear, tpr_linear, lw=1, alpha=0.3, label=fold_label)
        ax_linear_no_mean.plot(fpr_linear, tpr_linear, lw=1, alpha=0.3, label=fold_label)
        ax_linear_threshold_points.plot(fpr_linear, tpr_linear, lw=1, alpha=0.3, label=fold_label)
        ax_linear_no_mean_threshold_points.plot(fpr_linear, tpr_linear, lw=1, alpha=0.3, label=fold_label)
        
        ax_log.plot(fpr_log, tpr_log, lw=1, alpha=0.3, label=fold_label)
        ax_log_no_mean.plot(fpr_log, tpr_log, lw=1, alpha=0.3, label=fold_label)
        ax_log_threshold_points.plot(fpr_log, tpr_log, lw=1, alpha=0.3, label=fold_label)
        ax_log_no_mean_threshold_points.plot(fpr_log, tpr_log, lw=1, alpha=0.3, label=fold_label)
        
        ax_linear_threshold_points.scatter(fpr_linear, tpr_linear, s=10, label=fold_label)
        ax_linear_no_mean_threshold_points.scatter(fpr_linear, tpr_linear, s=10, label=fold_label)
        ax_log_threshold_points.scatter(fpr_log, tpr_log, s=10, label=fold_label)
        ax_log_no_mean_threshold_points.scatter(fpr_log, tpr_log, s=10, label=fold_label)
    
    # Plot mean ROC curves
    ax_linear.plot(mean_fpr_linear, mean_tpr_linear, color="b", label=f"Mean ROC (AUC = {mean_auc_linear:.2f} ± {std_auc_linear:.2f})", lw=2, alpha=0.8)
    
    ax_linear_threshold_points.plot(mean_fpr_linear, mean_tpr_linear, color="b", label=f"Mean ROC (AUC = {mean_auc_linear:.2f} ± {std_auc_linear:.2f})", lw=2, alpha=0.8)
    ax_linear_threshold_points.scatter(mean_fpr_linear, mean_tpr_linear, s=10, color="b")
    
    ax_log.set_xscale('log')
    ax_log.plot(mean_fpr_log, mean_tpr_log, color="b", label=f"Mean ROC (AUC = {mean_auc_log:.2f} ± {std_auc_log:.2f})", lw=2, alpha=0.8)
    ax_log.set_xlim(10**-5, 1)
    
    ax_log_threshold_points.set_xscale('log')
    ax_log_threshold_points.plot(mean_fpr_log, mean_tpr_log, color="b", label=f"Mean ROC (AUC = {mean_auc_log:.2f} ± {std_auc_log:.2f})", lw=2, alpha=0.8)
    ax_log_threshold_points.scatter(mean_fpr_log, mean_tpr_log, s=10, color="b")
    ax_log_threshold_points.set_xlim(10**-5, 1)
    
    ax_log_no_mean.set_xscale('log')
    ax_log_no_mean.set_xlim(10**-5, 1)

    ax_log_no_mean_threshold_points.set_xscale('log')
    ax_log_no_mean_threshold_points.set_xlim(10**-5, 1)
    
    # Set axis labels and titles
    for ax in [ax_linear, ax_linear_no_mean, ax_linear_threshold_points, ax_linear_no_mean_threshold_points,
               ax_log, ax_log_no_mean, ax_log_threshold_points, ax_log_no_mean_threshold_points]:
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="best")
    
    ax_linear.set_title("Mean ROC curve with variability (Linear Scale)")
    ax_linear_no_mean.set_title("Individual ROC curves (Linear Scale)")
    ax_log.set_title("Mean ROC curve with variability (Log Scale)")
    ax_log_no_mean.set_title("Individual ROC curves (Log Scale)")
    
    return (fig_linear, fig_linear_threshold_points, fig_linear_no_mean, fig_linear_no_mean_threshold_points,
            fig_log, fig_log_threshold_points, fig_log_no_mean, fig_log_no_mean_threshold_points)


def plot_multiple_roc_curves(mean_res, labels):
    # Initialize lists to store mean FPR and TPR values for each run, along with AUC scores
    mean_fpr_linear_list = []
    mean_tpr_linear_list = []
    auc_scores_linear = []
    
    mean_fpr_log_list = []
    mean_tpr_log_list = []
    auc_scores_log = []

    # Extract data for each run
    for res in mean_res:
        mean_fpr_linear_list.append(res['mean_fpr_linear'])
        mean_tpr_linear_list.append(res['mean_tpr_linear'])
        mean_fpr_log_list.append(res['mean_fpr_log'])
        mean_tpr_log_list.append(res['mean_tpr_log'])
        
        # Calculate AUC for linear and log scales
        auc_scores_linear.append(auc(res['mean_fpr_linear'], res['mean_tpr_linear']))
        auc_scores_log.append(auc(res['mean_fpr_log'], res['mean_tpr_log']))

    # Create the plots
    fig_linear, ax_linear = plt.subplots(figsize=(7, 6))
    fig_log, ax_log = plt.subplots(figsize=(7, 6))

    # Plot linear scale ROC curves
    for fpr, tpr, label, auc_score in zip(mean_fpr_linear_list, mean_tpr_linear_list, labels, auc_scores_linear):
        ax_linear.plot(fpr, tpr, label=f"{label} (AUC = {auc_score:.2f})")
    ax_linear.plot([0, 1], [0, 1], 'k--', lw=2, linestyle='--')
    ax_linear.set_title('ROC Curves (Linear Scale)')
    ax_linear.set_xlabel('False Positive Rate')
    ax_linear.set_ylabel('True Positive Rate')
    ax_linear.legend(loc="lower right")

    # Plot log scale ROC curves
    for fpr, tpr, label, auc_score in zip(mean_fpr_log_list, mean_tpr_log_list, labels, auc_scores_log):
        ax_log.plot(fpr, tpr, label=f"{label} (AUC = {auc_score:.2f})")
    ax_log.plot([0, 1], [0, 1], 'k--', lw=2, linestyle='--')
    ax_log.set_xscale('log')
    ax_log.set_title('ROC Curves (Log Scale)')
    ax_log.set_xlabel('False Positive Rate (Log Scale)')
    ax_log.set_ylabel('True Positive Rate')
    ax_log.legend(loc="lower right")

    plt.tight_layout()

    return fig_linear, fig_log