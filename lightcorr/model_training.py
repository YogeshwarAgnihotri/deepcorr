from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
from shared.utils import format_time

def train_model(model, training_data, labels):
    print("\nTraining the model...")
    start_time_training = time.time()
    model.fit(training_data, labels)
    training_time = time.time() - start_time_training

    print(f"Training completed in {format_time(training_time)}")
    return model

def train_classifier_halvingGridSearch(model, training_data, labels, param_grid, **kwargs):
    halving_grid_search = HalvingGridSearchCV(model, param_grid, **kwargs)
    return run_search_training(halving_grid_search, training_data, labels, 'halving_grid')

def train_classifier_gridSearch(model, training_data, labels, param_grid, **kwargs):
    # Define multiple scoring metrics, commented out for now
    # scoring = {
    #     'accuracy': make_scorer(accuracy_score),
    #     'precision': make_scorer(precision_score),
    #     'recall': make_scorer(recall_score),
    #     'f1_score': make_scorer(f1_score),
    #     'roc_auc_score': make_scorer(roc_auc_score)
    # }
    
    grid_search = GridSearchCV(model,param_grid, **kwargs)
    return run_search_training(grid_search, training_data, labels, 'grid')

def train_classifier_randomSearch(model, training_data, labels, param_distributions, **kwargs):
    # Define multiple scoring metrics, commented out for now
    # scoring = {
    #     'accuracy': make_scorer(accuracy_score),
    #     'precision': make_scorer(precision_score),
    #     'recall': make_scorer(recall_score),
    #     'f1_score': make_scorer(f1_score),
    #     'roc_auc_score': make_scorer(roc_auc_score)
    # }
    
    random_search = RandomizedSearchCV(model, param_distributions=param_distributions, **kwargs)
    return run_search_training(random_search, training_data, labels, 'random')


def run_search_training(search, training_data, labels, search_type):
    print(f"\nTraining the model with {search_type} search...")
    start_time = time.time()
    search.fit(training_data, labels)
    training_time = time.time() - start_time

    best_model = search.best_estimator_
    best_hyperparameters = search.best_params_
    best_score = search.best_score_
    cv_results = search.cv_results_

    print(f"\nTraining completed in {format_time(training_time)}")
    print(f"\nBest parameters: {best_hyperparameters}")
    print(f"\nMean cross-validated score of the best_estimator: {best_score}")

    return best_model, best_hyperparameters, cv_results