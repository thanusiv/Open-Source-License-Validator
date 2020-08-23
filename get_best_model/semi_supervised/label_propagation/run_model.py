#!/usr/bin/python3

import random
from sklearn.semi_supervised import LabelPropagation
import pandas as pd
import os
import pickle
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """
    This is a semi-supervised learning method called Label Propagation that can be used to attempt this task
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

    model_path = os.path.join(model_pickles_dir, 'label_propagation.pickle')
    confusion_matrix_path = os.path.join(model_confusion_matrix_dir, 'label_propagation_confusion_matrix.png')

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

    # mask 5% of the data
    y_train = mask_data(y_train.tolist(), 0.05)

    try:
        with open(model_path, 'rb') as data:
            license_classifier = pickle.load(data)
        print('Model was loaded in successfully!')
    except FileNotFoundError as e:
        print('Label Propagation model will begin training ...')
        license_classifier = LabelPropagation()
        license_classifier.fit(x_train.toarray(), y_train)
        print('Label Propagation model finished training!')

        print('Saving model ...')
        with open(model_path, 'wb') as output:
            pickle.dump(license_classifier, output)
        print('Saved!')

    print('Starting predictions ...')
    train_predictions = license_classifier.predict(x_train)
    test_predictions = license_classifier.predict(x_test)
    print('Predictions complete!')

    # Training accuracy
    print("The training accuracy is: ")
    print(accuracy_score(y_train, train_predictions))

    # Test accuracy
    print("The test accuracy is: ")
    print(accuracy_score(y_test, test_predictions))

    # Classification report
    print("Classification report")
    print(classification_report(y_validation, test_predictions))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_validation, test_predictions)
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


def mask_data(y, percentage):
    """
    Code used to mask the data. A -1 for a label means that the label is unknown

    :param y: labels
    :param percentage: the percentage of which the data should be masked

    :return: the new labels with masked data
    :rtype: list
    """
    new_y = []
    for value in y:
        change = random.random()
        if change < percentage:
            new_y.append(-1)
        else:
            new_y.append(value)
    return new_y


if __name__ == '__main__':
    main()
