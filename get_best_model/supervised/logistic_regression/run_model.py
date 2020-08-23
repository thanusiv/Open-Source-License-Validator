#!/usr/bin/python3
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, ShuffleSplit
from pprint import pprint
import numpy as np
import pandas as pd
from scipy import sparse


def main():
    """
    Using the Logistic Regression model here to predict licenses using Random and Grid Search for cross-validation and
    hyperparameter tuning
    """
    os.chdir('../../../all_files_generated')
    current_dir = os.getcwd()

    data_pickles_dir = os.path.join(current_dir, 'data_pickles')
    model_pickles_dir = os.path.join(current_dir, 'model_pickles')
    model_confusion_matrix_dir = os.path.join(current_dir, 'model_confusion_matrix_files')

    x_train_path = os.path.join(data_pickles_dir, 'x_train.pickle')
    x_validation_path = os.path.join(data_pickles_dir, 'x_validation.pickle')
    x_test_path = os.path.join(data_pickles_dir, 'x_test.pickle')
    y_train_path = os.path.join(data_pickles_dir, 'y_train.pickle')
    y_validation_path = os.path.join(data_pickles_dir, 'y_validation.pickle')
    y_test_path = os.path.join(data_pickles_dir, 'y_test.pickle')

    model_path = os.path.join(model_pickles_dir, 'logistic_regression.pickle')
    confusion_matrix_path = os.path.join(model_confusion_matrix_dir, 'logistic_regression_confusion_matrix.png')

    # read in all pickle files that may be required
    with open(x_train_path, 'rb') as data:
        x_train = pickle.load(data)

    with open(x_validation_path, 'rb') as data:
        x_validation = pickle.load(data)

    with open(x_test_path, 'rb') as data:
        x_test = pickle.load(data)

    with open(y_train_path, 'rb') as data:
        y_train = pickle.load(data)

    with open(y_validation_path, 'rb') as data:
        y_validation = pickle.load(data)

    with open(y_test_path, 'rb') as data:
        y_test = pickle.load(data)

    # combine training and validation datasets
    x_train = sparse.vstack((x_train, x_validation))  # scipy.sparse.csr matrix
    y_train = y_train.append(pd.Series(y_validation))  # pandas series

    try:
        with open(model_path, 'rb') as data:
            best_license_classifier = pickle.load(data)
        print('Model was loaded in successfully!')
    except FileNotFoundError as e:
        print('Logistic Regression model will begin training ...')
        license_classifier = LogisticRegression(random_state=8)

        print('Parameters that can be used:\n')
        pprint(license_classifier.get_params())

        # C = Inverse of regularization strength. Smaller values specify stronger regularization.
        # multi_class = can use multinomial or ovr
        # solver = Algorithm to use in the optimization problem. For multiclass problems, only newton-cg, sag, saga and lbfgs handle multinomial loss.
        # class_weight: Weights associated with classes.
        # penalty: Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.

        # C
        C = [float(x) for x in np.linspace(start=0.1, stop=1, num=10)]

        # multi_class
        multi_class = ['ovr', 'multinomial']

        # solver
        solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

        # class_weight
        class_weight = ['balanced', None]

        # penalty
        penalty = ['l1', 'l2', 'elasticnet', 'none']

        # Create the random grid
        random_grid = {'C': C,
                       'multi_class': multi_class,
                       'solver': solver,
                       'class_weight': class_weight,
                       'penalty': penalty}

        # Definition of the random search
        random_search = RandomizedSearchCV(estimator=license_classifier,
                                           param_distributions=random_grid,
                                           n_iter=200,
                                           scoring='accuracy',
                                           cv=3,
                                           verbose=1,
                                           random_state=8)

        # Fit the random search model
        random_search.fit(x_train, y_train)

        print("The best hyperparameters from Random Search are:")
        print(random_search.best_params_)
        print("")
        print("The mean accuracy of a model with these hyperparameters is:")
        print(random_search.best_score_)

        # Create the parameter grid based on the results of random search
        C = [float(x) for x in
             np.linspace(start=random_search.best_params_['C'] - 0.2, stop=random_search.best_params_['C'] + 0.2,
                         num=10)]
        multi_class = [random_search.best_params_['multi_class']]
        solver = [random_search.best_params_['solver']]
        class_weight = [random_search.best_params_['class_weight']]
        penalty = [random_search.best_params_['penalty']]

        param_grid = {'C': C,
                      'multi_class': multi_class,
                      'solver': solver,
                      'class_weight': class_weight,
                      'penalty': penalty}

        # Manually create the splits in CV
        cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=license_classifier,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   cv=cv_sets,
                                   verbose=1)

        # Fit the grid search to the data
        grid_search.fit(x_train, y_train)

        print("The best hyperparameters from Grid Search are:")
        print(grid_search.best_params_)
        print("")
        print("The mean accuracy of a model with these hyperparameters is:")
        print(grid_search.best_score_)

        best_license_classifier = grid_search.best_estimator_
        best_license_classifier.fit(x_train, y_train)
        print('Logistic Regression model finished training!')

        print('Saving model ...')
        with open(model_path, 'wb') as output:
            pickle.dump(best_license_classifier, output)
        print('Saved!')

    print('Starting predictions ...')
    train_predictions = best_license_classifier.predict(x_train)
    test_predictions = best_license_classifier.predict(x_test)
    print('Predictions complete!')

    # Training accuracy
    print("The training accuracy is: ")
    print(accuracy_score(y_train, train_predictions))

    # Test accuracy
    print("The test accuracy is: ")
    print(accuracy_score(y_test, test_predictions))

    # Classification report
    print("Classification report")
    print(classification_report(y_test, test_predictions))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, test_predictions)
    print('Confusion Matrix')
    print(conf_matrix)
    plt.figure(figsize=(12.8, 6))
    sns.heatmap(conf_matrix,
                annot=True,
                xticklabels=['not_license', 'license'],
                yticklabels=['not_license', 'license'],
                cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    plt.savefig(confusion_matrix_path)
    plt.show()


if __name__ == '__main__':
    main()
