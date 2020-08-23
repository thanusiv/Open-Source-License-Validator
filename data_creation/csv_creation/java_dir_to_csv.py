import os
import csv


def main():
    """
    This file creates a csv file by going through each directory in java_files, placing a specific label for it (1 or 0)
    and then writing it to a csv file
    """
    current_dir = os.getcwd()
    cat_directory = os.path.join(current_dir, 'java_files')
    os.chdir(cat_directory)
    values = []

    for directory in os.listdir():
        if directory == 'license':
            label = 1
        else:
            label = 0
        directory_path = os.path.join(cat_directory, directory)
        os.chdir(directory_path)
        for txt_file in os.listdir():
            with open(directory_path + '/' + txt_file, 'r', encoding='utf-8') as file:
                data = file.read()
            values.append({'comment_block_text': data, 'label': label})

    os.chdir('../../../../all_files_generated/csv_files')
    current_dir = os.getcwd()
    file_dir = os.path.join(current_dir, 'data_java.csv')
    fields = ['comment_block_text', 'label']

    # write to csv file
    with open(file_dir, 'w', encoding='utf-8', newline='') as output_csv:
        output_writer = csv.DictWriter(output_csv, fieldnames=fields)
        output_writer.writeheader()
        for item in values:
            output_writer.writerow(item)


if __name__ == '__main__':
    main()
