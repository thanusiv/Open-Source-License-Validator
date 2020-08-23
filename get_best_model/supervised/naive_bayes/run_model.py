#!/usr/bin/python3
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import sparse
import pandas as pd


def main():
    """
    Using the Naive Bayes model to predict licenses. No hyperparameter tuning was used since the only parameter that
    could be tuned would be smoothing
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

    model_path = os.path.join(model_pickles_dir, 'naive_bayes.pickle')
    confusion_matrix_path = os.path.join(model_confusion_matrix_dir, 'naive_bayes_confusion_matrix.png')

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
            license_classifier = pickle.load(data)
        print('Model was loaded in successfully!')
    except FileNotFoundError as e:
        print('Naive Bayes model will begin training ...')
        license_classifier = MultinomialNB()
        license_classifier.fit(x_train, y_train)
        print('Naive Bayes model finished training!')

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
