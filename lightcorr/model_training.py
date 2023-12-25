from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from shared.utils import format_time

def train_classifier_gridSearch(model, training_data, labels, param_grid, cross_validation_folds, verbosity):
    grid_search = GridSearchCV(model, param_grid, cv=cross_validation_folds, scoring='accuracy', verbose=verbosity, n_jobs=-1)

    print("\nTraining the model with grid search...")
    start_time_grid_search = time.time()
    grid_search.fit(training_data, labels)
    training_time_grid_search = time.time() - start_time_grid_search

    best_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_

    print(f"Training completed in {format_time(training_time_grid_search)}")
    print(f"Best parameters: {best_parameters}")

    return best_model, best_parameters

def train_classifier_randomSearch(model, training_data, labels, param_distributions, n_iter=10, cross_validation_folds=5, scoring=None, n_jobs=-1, refit=True, cv=None, verbosity=2, pre_dispatch='2*n_jobs', random_state=None, error_score='raise', return_train_score=False):
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv or cross_validation_folds, verbose=verbosity, pre_dispatch=pre_dispatch, random_state=random_state, error_score=error_score, return_train_score=return_train_score)

    print("\nTraining the model with random search...")
    start_time_random_search = time.time()
    random_search.fit(training_data, labels)
    training_time_random_search = time.time() - start_time_random_search

    best_model = random_search.best_estimator_
    best_parameters = random_search.best_params_

    print(f"Training completed in {format_time(training_time_random_search)}")
    print(f"Best parameters: {best_parameters}")

    return best_model, best_parameters

def train_model(model, training_data, labels):
    print("\nTraining the model...")
    start_time_training = time.time()
    model.fit(training_data, labels)
    training_time = time.time() - start_time_training

    print(f"Training completed in {format_time(training_time)}")
    return model
