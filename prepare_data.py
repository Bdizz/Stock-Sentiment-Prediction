import csv
import os
import random


def data_prep():

    # load the labeled data from the CSV
    with open(os.path.join(os.path.dirname(__file__), 'data/labeled_data.csv'), newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)

    # shuffle the data
    random.shuffle(data)

    # determine the split point between data for train.csv and test.csv
    # this will give an 80/20 split
    split_index = int(len(data) * .8)

    # split the data into train and test
    train_data = data[:split_index]
    test_data = data[split_index:]

    # write the training and testing data to the csv's
    with open(os.path.join(os.path.dirname(__file__), 'data/train.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(train_data)

    with open(os.path.join(os.path.dirname(__file__), 'data/test.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(test_data)

    print(len(train_data))
    print(len(test_data))