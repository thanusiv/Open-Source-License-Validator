#!/usr/bin/python3
import os
import pickle
from scipy import sparse
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.pairwise import cosine_similarity


def main():
    """
    An attempt to use an autoencoder to predict licenses.
    """
    os.chdir('../../../all_files_generated')
    current_dir = os.getcwd()

    data_pickles_dir = os.path.join(current_dir, 'data_pickles')
    model_pickles_dir = os.path.join(current_dir, 'model_pickles')

    x_train_path = os.path.join(data_pickles_dir, 'x_train.pickle')
    x_validation_path = os.path.join(data_pickles_dir, 'x_validation.pickle')
    x_test_path = os.path.join(data_pickles_dir, 'x_test.pickle')
    y_train_path = os.path.join(data_pickles_dir, 'y_train.pickle')
    y_validation_path = os.path.join(data_pickles_dir, 'y_validation.pickle')
    y_test_path = os.path.join(data_pickles_dir, 'y_test.pickle')

    model_path = os.path.join(model_pickles_dir, 'autoencoder.pickle')

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

    # combine all datasets
    x_train = sparse.vstack((x_train, x_validation, x_test)).todense()  # <class 'numpy.matrix'>
    y_train = y_train.append(pd.Series(y_validation))  # pandas series
    y_train = y_train.append(pd.Series(y_test))  # pandas series

    x_train_0 = []
    x_train_1 = []
    y_train_0 = []
    y_train_1 = []

    for x, y in zip(x_train, y_train):
        if y == 0:
            x_train_0.append(x)
            y_train_0.append(y)
        else:
            x_train_1.append(x)
            y_train_1.append(y)

    x_train_0 = np.array(x_train_0)
    x_train_0 = x_train_0[:, 0, :]

    x_train_1 = np.array(x_train_1)
    x_train_1 = x_train_1[:, 0, :]

    try:
        with open(model_path, 'rb') as data:
            license_classifier = pickle.load(data)
        print('Model was loaded in successfully!')
    except FileNotFoundError as e:
        print('Autoencoder model will begin training ...')
        license_classifier = MLPRegressor(hidden_layer_sizes=(500, 125, 500))
        license_classifier.fit(x_train_1, x_train_1)
        print('Autoencoder model finished training!')

        print('Saving model ...')
        with open(model_path, 'wb') as output:
            pickle.dump(license_classifier, output)
        print('Saved!')

    print('Starting predictions ...')
    train_predictions = license_classifier.predict(x_train_1)
    validation_predictions = license_classifier.predict(x_train_0)
    print('Predictions complete!')

    # Training accuracy
    print("The training accuracy is: ")
    print(license_classifier.score(x_train_1, train_predictions))

    # Test accuracy (should be low since the model should have a hard time recreating invalid licenses)
    print("The test accuracy is: ")
    print(license_classifier.score(x_train_0, validation_predictions))

    # can change the following two lines based on what you need
    sorted_cosine_similarities_1 = get_computed_similarities(x_train_1, train_predictions, y_train_1,
                                                             True)
    sorted_cosine_similarities_0 = get_computed_similarities(x_train_0, validation_predictions, y_train_0,
                                                             True)

    display_most_or_least_similar(sorted_cosine_similarities_1, 10, True)
    display_most_or_least_similar(sorted_cosine_similarities_0, 10, True)


def get_computed_similarities(original_vectors, predicted_vectors, labels, descending=False):
    data_size = len(original_vectors)
    cosine_similarities = []
    for i in range(data_size):
        cosine_sim_val = cosine_similarity(original_vectors[i], predicted_vectors[i])
        cosine_similarities.append((i, cosine_sim_val, labels[i]))
    return sorted(cosine_similarities, key=lambda x: x[1], reverse=descending)


def display_most_or_least_similar(sorted_cosine_similarities, n, descending):
    if descending:
        print('Most similar documents\' values and their actual label')
    else:
        print('Least similar documents\' values and their actual label')

    for i in range(n):
        print('Value:', sorted_cosine_similarities[i][1], 'Label:', sorted_cosine_similarities[i][2])


if __name__ == '__main__':
    main()
