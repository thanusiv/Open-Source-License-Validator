import pandas as pd
from preprocessor import Preprocessor
from essential_generators import DocumentGenerator
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import random


def main():
    """
    This file creates the pickle files that will be used by all the models and the text files for fastText. The
    preprocessing is handled here as well using the preprocessor class.
    """
    os.chdir('../../all_files_generated/csv_files')
    current_dir = os.getcwd()
    data_java_csv_path = os.path.join(current_dir, 'data_java.csv')
    open_source_licenses_path = os.path.join(current_dir, 'open_source_licenses.csv')

    data_java_df = pd.read_csv(data_java_csv_path)
    data_java_df.drop_duplicates(inplace=True)

    open_source_df = pd.read_csv(open_source_licenses_path)
    open_source_df.drop_duplicates(inplace=True)
    open_source_df['label'] = open_source_df['license_type'].apply(lambda x: 0 if x == 'INVALID' else 1)
    open_source_df = open_source_df[['comment_block_text', 'label']]

    # generate some new comment blocks that contain some ninka text so that the model can differentiate that just
    # because a specific word is in the comment doesn't mean it contains a license
    gen = DocumentGenerator()
    number_of_generated_blocks = 1000
    generated_data = [gen.paragraph(min_sentences=1) for i in range(number_of_generated_blocks)]
    generated_data_labels = [0 for i in range(number_of_generated_blocks)]

    with open(os.path.join(os.path.dirname(__file__) + '/words_for_generation.dict'), 'r') as f:
        license_keywords = [x.strip().lower() for x in
                            filter(lambda x: not x.startswith("#") and x, f.readlines())]

    updated_generated_data = []
    for x in generated_data:
        number_of_select_words = random.randint(0, 15)
        select_words = []

        for _ in range(number_of_select_words):
            select_words.append(random.choice(license_keywords))

        for word in select_words:
            insertion_index = random.randint(0, len(x) - 1)
            x = x[:insertion_index] + ' ' + word + ' ' + x[insertion_index:]

        updated_generated_data.append(x)

    generated_data_df = pd.DataFrame({'comment_block_text': updated_generated_data, 'label': generated_data_labels})
    combined_df = pd.concat([data_java_df, open_source_df, generated_data_df], ignore_index=True).drop_duplicates(
        keep='first')
    combined_df = combined_df.append(combined_df[combined_df.label == 1].sample(n=3000)) # comment this out if you do not wish to have oversampling

    print('Number of Invalid Comment Blocks (0):', str(len(combined_df[combined_df.label == 0])))
    print('number of Valid Comment Blocks (1):', str(len(combined_df[combined_df.label == 1])))

    preprocessor = Preprocessor()
    print('Start preprocessing ...')
    combined_df['clean_text'] = combined_df['comment_block_text'].apply(
        lambda n: preprocessor.preprocess(str(n)))
    open_source_df['clean_text'] = open_source_df['comment_block_text'].apply(lambda n: preprocessor.preprocess(str(n)))
    print('Finish preprocessing')

    # 60% train, 20% validation, 20% test
    x_temp, x_test, y_temp, y_test = train_test_split(combined_df['clean_text'], combined_df['label'],
                                                      train_size=0.8, test_size=0.2, random_state=8)
    x_train, x_validation, y_train, y_validation = train_test_split(x_temp, y_temp,
                                                                    train_size=0.75, test_size=0.25, random_state=8)

    os.chdir("../../all_files_generated/text_files")
    current_dir = os.getcwd()

    train_text_file_dir = os.path.join(current_dir, 'train.txt')
    validation_text_file_dir = os.path.join(current_dir, 'validation.txt')
    train_validation_text_file_dir = os.path.join(current_dir, 'train_validation.txt')
    test_text_file_dir = os.path.join(current_dir, 'test.txt')

    print('Creating text files for the fastText model ...')

    print('Start writing to train.txt ...')
    with open(train_text_file_dir, 'w', encoding='utf-8') as training_file:
        for x, y in zip(x_train, y_train):
            # print(x + ' __label__' + str(y))
            training_file.write(x + ' __label__' + str(y) + '\n')
            # print('-------------')
    print('Finished writing to train.txt')

    print('Start writing to validation.txt ...')
    with open(validation_text_file_dir, 'w', encoding='utf-8') as validation_file:
        for x, y in zip(x_validation, y_validation):
            # print(x + ' __label__' + str(y))
            validation_file.write(x + ' __label__' + str(y) + '\n')
            # print('-------------')

    print('Finished writing to validation.txt')

    print('Start writing to test.txt ...')
    with open(test_text_file_dir, 'w', encoding='utf-8') as test_file:
        for x, y in zip(x_validation, y_validation):
            # print(x + ' __label__' + str(y))
            test_file.write(x + ' __label__' + str(y) + '\n')
            # print('-------------')

    print('Finished writing to test.txt')

    print('Start writing to train_validation.txt ...')
    with open(train_validation_text_file_dir, 'w', encoding='utf-8') as train_validation_file:
        for x, y in zip(x_train, y_train):
            train_validation_file.write(x + ' __label__' + str(y) + '\n')
        for x, y in zip(x_validation, y_validation):
            train_validation_file.write(x + ' __label__' + str(y) + '\n')

    print('Finished writing to train_validation.txt')

    print('Finished creating the text files. They will be found in all_files_generated/text_files')

    # Parameter selection for TFIDF
    ngram_range = (1, 2)
    min_df = 10
    max_df = 1.
    max_features = 300
    category = {'not_license': 0, 'license': 1}

    # using some default parameters here. Change this to whatever you like using the parameters above
    vectorizer = TfidfVectorizer()
    x_train_vectors = vectorizer.fit_transform(x_train)
    x_validation_vectors = vectorizer.transform(x_validation)
    x_test_vectors = vectorizer.transform(x_test)

    os.chdir("../../all_files_generated/data_pickles")
    current_dir = os.getcwd()

    x_train_dir = os.path.join(current_dir, 'x_train.pickle')
    x_validation_dir = os.path.join(current_dir, 'x_validation.pickle')
    x_test_dir = os.path.join(current_dir, 'x_test.pickle')
    y_train_dir = os.path.join(current_dir, 'y_train.pickle')
    y_validation_dir = os.path.join(current_dir, 'y_validation.pickle')
    y_test_dir = os.path.join(current_dir, 'y_test.pickle')

    df_dir = os.path.join(current_dir, 'df.pickle')

    print('Creating pickle files ...')

    # x_train
    with open(x_train_dir, 'wb') as output:
        pickle.dump(x_train_vectors, output)

    # x_validation
    with open(x_validation_dir, 'wb') as output:
        pickle.dump(x_validation_vectors, output)

    # x_test
    with open(x_test_dir, 'wb') as output:
        pickle.dump(x_test_vectors, output)

    # y_train
    with open(y_train_dir, 'wb') as output:
        pickle.dump(y_train, output)

    # y_validation
    with open(y_validation_dir, 'wb') as output:
        pickle.dump(y_validation, output)

    # y_test
    with open(y_test_dir, 'wb') as output:
        pickle.dump(y_test, output)

    # df
    with open(df_dir, 'wb') as output:
        pickle.dump(combined_df, output)

    print('Finished creation of pickle files. They will be found in all_files_generated/data_pickles.')


if __name__ == '__main__':
    main()
