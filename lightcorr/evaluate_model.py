from lightcorr.model_validation import evaluate_test_set

def main():
    # Evaluate the model on the test set
    evaluate_test_set(model, flattend_flow_pairs_test, flattend_labels_test)

if __name__ == "__main__":
    main()

# ######################################Evaluation Settings######################################
# evaluation_settings:
#   roc_plot_enabled: false  # If true, generate ROC plot   # TODO IMPLEMENT roc with test
#   evaluate_on_test_set: false  # If true, evaluate the model on the test set