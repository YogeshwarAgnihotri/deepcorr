# evaluation.py
# Functions for evaluating the trained model and printing metrics.
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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