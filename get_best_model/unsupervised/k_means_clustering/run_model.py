#!/usr/bin/python3
import os
import pickle
from sklearn.cluster import KMeans
from scipy import sparse
import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer


def main():
    """
    Using k-means for some data exploration and a potential solution for the license prediction problem
    """
    os.chdir('../../../all_files_generated')
    current_dir = os.getcwd()

    data_pickles_dir = os.path.join(current_dir, 'data_pickles')
    elbow_method_files_dir = os.path.join(current_dir, 'elbow_method_files')

    x_train_path = os.path.join(data_pickles_dir, 'x_train.pickle')
    x_validation_path = os.path.join(data_pickles_dir, 'x_validation.pickle')
    x_test_path = os.path.join(data_pickles_dir, 'x_test.pickle')
    y_train_path = os.path.join(data_pickles_dir, 'y_train.pickle')
    y_validation_path = os.path.join(data_pickles_dir, 'y_validation.pickle')
    y_test_path = os.path.join(data_pickles_dir, 'y_test.pickle')

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
    x_train = sparse.vstack((x_train, x_validation, x_test))  # scipy.sparse.csr matrix
    y_train = y_train.append(pd.Series(y_validation))  # pandas series
    y_train = y_train.append(pd.Series(y_test))  # pandas series

    use_yellowbrick = False

    if use_yellowbrick:
        license_classifier = KMeans()
        visualizer = KElbowVisualizer(license_classifier, k=(2, 100))
        visualizer.fit(x_train)
        visualizer.show()
    else:
        inertia = []
        k = range(2, 100)
        for i in k:
            license_classifier = KMeans(n_clusters=i)
            license_classifier.fit(x_train)
            inertia.append(license_classifier.inertia_)

        plt.plot(k, inertia)
        plt.xlabel('K')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')

        elbow_method_path = os.path.join(elbow_method_files_dir, 'k_means_clustering_elbow_method.png')
        plt.savefig(elbow_method_path)

        plt.show()


if __name__ == '__main__':
    main()
