# model_training.py
# Functions for training the decision tree model.
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
import datetime
from shared.utils import format_time

def dt_train_classifier_gridSearch(training_data, labels, param_grid, cross_validation_folds, verbosity):
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=cross_validation_folds, scoring='accuracy', verbose=verbosity, n_jobs=-1)

    # Start timing the training process
    start_time_grid_search = time.time()

    # do grid seach and make predictions with best model
    print("\nTraining the model with grid search of tree hyperparameters...")
    grid_search.fit(training_data, labels)
    best_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_
    # End timing the training process with grid search
    training_time_grid_search = time.time() - start_time_grid_search
    print(f"Training completed in {format_time(training_time_grid_search)}")

    print(f"Best parameters: {best_parameters}")

    return best_model, best_parameters

def dt_train_classifier_randomSearch(training_data, labels, param_distributions, n_iter=10, cross_validation_folds=5, scoring=None, n_jobs=-1, refit=True, cv=None, verbosity=2, pre_dispatch='2*n_jobs', random_state=None, error_score='raise', return_train_score=False):
    random_search = RandomizedSearchCV(
        DecisionTreeClassifier(),
        param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
        cv=cv or cross_validation_folds,  # Use 'cv' if provided, else default to 'cross_validation_folds'
        verbose=verbosity,
        pre_dispatch=pre_dispatch,
        random_state=random_state,
        error_score=error_score,
        return_train_score=return_train_score
    )

    # Start timing the training process
    start_time_random_search = time.time()

    # Perform random search and fit the model
    print("\nTraining the model with random search of tree hyperparameters...")
    random_search.fit(training_data, labels)
    best_model = random_search.best_estimator_
    best_parameters = random_search.best_params_

    # End timing the training process with random search
    training_time_random_search = time.time() - start_time_random_search
    print(f"Training completed in {format_time(training_time_random_search)}")
    print(f"Best parameters: {best_parameters}")

    return best_model, best_parameters
def dt_train(training_data, 
             labels,
             criterion="gini", 
             splitter="best", 
             max_depth=None, 
             min_samples_split=2, 
             min_samples_leaf=1, 
             min_weight_fraction_leaf=0.0, 
             max_features=None, 
             random_state=None, 
             max_leaf_nodes=None, 
             min_impurity_decrease=0.0, 
             class_weight=None, 
             ccp_alpha=0.0):

    # Create a Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion=criterion, 
                                 splitter=splitter, 
                                 max_depth=max_depth, 
                                 min_samples_split=min_samples_split, 
                                 min_samples_leaf=min_samples_leaf, 
                                 min_weight_fraction_leaf=min_weight_fraction_leaf, 
                                 max_features=max_features, 
                                 random_state=random_state, 
                                 max_leaf_nodes=max_leaf_nodes,
                                 min_impurity_decrease=min_impurity_decrease, 
                                 class_weight=class_weight, 
                                 ccp_alpha=ccp_alpha)
    # Start timing the training process
    start_time_traning = time.time()
    print("\nTraining the decision tree...")
    # Train the model
    model = clf.fit(training_data, labels)
    # End timing the training process
    training_time = time.time() - start_time_traning
    print(f"Training of decision tree completed in {format_time(training_time)}")

    return model