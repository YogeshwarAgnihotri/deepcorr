from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from shared.utils import format_time

def train_model(model, training_data, labels):
    print("\nTraining the model...")
    start_time_training = time.time()
    model.fit(training_data, labels)
    training_time = time.time() - start_time_training

    print(f"Training completed in {format_time(training_time)}")
    return model

def train_classifier_gridSearch(model, training_data, labels, param_grid, **kwargs):
    grid_search = GridSearchCV(model, param_grid, **kwargs)
    return run_search_training(grid_search, training_data, labels, 'grid')

def train_classifier_randomSearch(model, training_data, labels, param_distributions, **kwargs):
    random_search = RandomizedSearchCV(model, param_distributions, **kwargs)
    return run_search_training(random_search, training_data, labels, 'random')


def run_search_training(search, training_data, labels, search_type):
    print(f"\nTraining the model with {search_type} search...")
    start_time = time.time()
    search.fit(training_data, labels)
    training_time = time.time() - start_time

    best_model = search.best_estimator_
    best_hyperparameters = search.best_params_
    best_score = search.best_score_

    print(f"\nTraining completed in {format_time(training_time)}")
    print(f"\nBest parameters: {best_hyperparameters}")
    print(f"\nMean cross-validated score of the best_estimator: {best_score}")

    return best_model, best_hyperparameters

