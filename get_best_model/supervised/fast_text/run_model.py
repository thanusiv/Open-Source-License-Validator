#!/usr/bin/python3

import fasttext
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    """
    Using the fastText model here to predict licenses using automatic hyperparameter tuning
    """
    os.chdir('../../../all_files_generated')
    current_dir = os.getcwd()

    text_files_dir = os.path.join(current_dir, 'text_files')
    model_pickles_dir = os.path.join(current_dir, 'model_pickles')
    model_confusion_matrix_dir = os.path.join(current_dir, 'model_confusion_matrix_files')

    training_validation_file_path = os.path.join(text_files_dir, 'train_validation.txt')
    test_file_path = os.path.join(text_files_dir, 'test.txt')

    model_path = os.path.join(model_pickles_dir, 'fasttext.pickle')
    confusion_matrix_path = os.path.join(model_confusion_matrix_dir, 'fast_text_confusion_matrix.png')

    try:
        license_classifier = fasttext.load_model(model_path)
        print('Model was loaded in successfully!')
    except ValueError as e:
        print('fastText model will begin training ...')
        license_classifier = fasttext.train_supervised(input=training_validation_file_path,
                                                       autotuneValidationFile=test_file_path,
                                                       autotuneDuration=60)
        print('fastText model finished training')
        print('Saving model ...')
        license_classifier.save_model(model_path)
        print('Saved!')

    print('Starting predictions ...')

    x_train = []
    y_train = []
    train_predictions = []
    with open(training_validation_file_path, 'r', encoding='utf-8') as train_file:
        for line in train_file.readlines():
            line_array = line.split('__label__')
            comment_block_text = line_array[0].strip()
            label = int(line_array[1])
            x_train.append(comment_block_text)
            y_train.append(label)
            train_predictions.append(int(license_classifier.predict(comment_block_text)[0][0][9:]))

    x_test = []
    test_predictions = []
    y_test = []
    with open(test_file_path, 'r', encoding='utf-8') as validation_file:
        for line in validation_file.readlines():
            line_array = line.split('__label__')
            comment_block_text = line_array[0].strip()
            label = int(line_array[1])
            x_test.append(comment_block_text)
            y_test.append(label)
            test_predictions.append(int(license_classifier.predict(comment_block_text)[0][0][9:]))

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
