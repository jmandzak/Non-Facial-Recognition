import pandas as pd
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
import tensorflow_addons as tfa
from sklearn import svm
from sklearn.metrics import classification_report
import sys

train = "cities_dataset_10/train_validation"
test = "cities_dataset_10/test"

def make_df(filename):
    cities = []
    files = []
    for city in os.listdir(filename):
        d = os.path.join(filename, city)
        if os.path.isdir(d):
            for pic in os.listdir(d):
                name = os.path.join(d, pic)
                cities.append(city)
                files.append(name)

    
    dic = {'City': cities, 'File': files}
    return pd.DataFrame(dic, columns=['City', 'File'])

def evaluate_embeddings(model, test_images, test_labels, is_baseline):
    if is_baseline:
        # get the baseline model, remove the softmax layer
        final_model = Sequential()
        for layer in model.layers[:-1]:
            final_model.add(layer)
    else:
        final_model = model

    y_pred = final_model.predict(test_images)

    true_match = 0
    false_match = 0
    true_different = 0
    false_different = 0

    # normalize vector
    y_pred = y_pred / np.max(y_pred)

    # we need to find the best split for both the baseline and the trip loss model
    split = 0.5
    labels = list(test_labels)
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            l2 = np.sum(np.power((y_pred[i]-y_pred[j]),2))
            # first check to see if the labels are the same
            if labels[i] == labels[j]:
                if l2 < split:
                    true_match += 1
                else:
                    false_different += 1
            # this means they're different
            else:
                if l2 < split:
                    false_match += 1
                else:
                    true_different += 1

    print(f'true_match: {true_match}')
    print(f'true_different: {true_different}')
    print()
    print(f'false_match: {false_match}')
    print(f'false_different: {false_different}')
    print(f'accuracy: {(true_match+true_different)/(false_match+false_different+true_match+true_different)}')

def main():

    # first read in the data if we need to
    new_read = not os.path.exists('train_images.npy')
    if new_read:
        train_df = make_df(train)
        test_df = make_df(test)

        # create master list
        train_images = np.zeros((8477, 112, 112, 3))

        # read in images
        for i, filename in enumerate(train_df.File):
            image = cv2.imread(filename)
            # normalize and resize the image
            image = image / 255
            image = cv2.resize(image, (112, 112))

            train_images[i] = image

        test_images = np.zeros((1400, 112, 112, 3))
        for i, filename in enumerate(test_df.File):
            image = cv2.imread(filename)
            # normalize and resize the image
            image = image / 255
            image = cv2.resize(image, (112, 112))

            test_images[i] = image

        # convert python lists to numpy arrays
        train_images = np.asarray(train_images)
        test_images = np.asarray(test_images)

        # write the results to files so we don't have to reread everything
        np.save('train_images', train_images)
        np.save('test_images', test_images)
        train_df.to_csv('train_df.csv')
        test_df.to_csv('test_df.csv')
    
    else:
        train_images = np.load('train_images.npy')
        test_images = np.load('test_images.npy')
        train_df = pd.read_csv('train_df.csv')
        test_df = pd.read_csv('test_df.csv')
    
    # get the test data for zero shot testing
    train_images_holdout = train_images[:1694]
    train_df_holdout = train_df.iloc[:1694, :]

    if len(sys.argv) < 2:
        print('Usage: python test_model.py [model_file_name]')
        exit()

    model_file_name = sys.argv[1]
    type_model = ''
    if 'baseline' in model_file_name:
        type_model = 'cce'
    else:
        type_model = 'trip'

    type_data = ''
    if 'holdout' in model_file_name:
        type_data = 'holdout'
    else:
        type_data = 'full'

    model = tf.keras.models.load_model(model_file_name)
    if type_model == 'cce':
        if type_data == 'full':
            encoded_test_labels = pd.get_dummies(test_df.City)
            print(model.evaluate(test_images, encoded_test_labels))
        else:
            evaluate_embeddings(model, train_images_holdout, train_df_holdout.City, True)
    else:
        if type_data == 'full':
            numerical_test_labels = (test_df.City).replace(['Amsterdam', 'Barcelona', 'Bucharest', 'Budapest', 'Istanbul', 'London', 'Paris', 'Rome', 'Stockholm', 'Vienna'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            test_embeddings = model.predict(test_images)

            # create svm to classify embeddings
            clf = svm.SVC()
            clf.fit(test_embeddings, numerical_test_labels)
            y_pred = clf.predict(test_embeddings)
            print(classification_report(numerical_test_labels, y_pred, digits=4))
        else:
            evaluate_embeddings(model, train_images_holdout, train_df_holdout.City, False)


if __name__ == '__main__':
    main()