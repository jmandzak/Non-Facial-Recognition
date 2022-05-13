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

def create_base_model(train_images, train_labels, test_images, test_labels):
    # custom made CNN
    model = Sequential()
    model.add(layers.Conv2D(32, (4,4), input_shape=((train_images[0].shape)), activation='relu', padding='same'))
    # model.add(layers.Conv2D(32, (4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(4,4),strides=(4,4)))
    # model.add(layers.Conv2D(64, (4,4), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(4,4),strides=(4,4)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(10, activation='softmax'))
    model.add(layers.Dense(8, activation='softmax'))

    lr = .001
    optimizer = optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    encoded_train_labels = pd.get_dummies(train_labels)
    encoded_test_labels = pd.get_dummies(test_labels)
    
    # early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # create tensorboard
    #creating unique name for tensorboard directory
    log_dir = "logs/" + str(f'holdout-baseline-opt=Adagrad-128')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(train_images, encoded_train_labels, epochs=100, validation_data=(test_images, encoded_test_labels), callbacks=[tensorboard_callback, es])
    # model.fit(train_images, encoded_train_labels, epochs=5, validation_data=(test_images, encoded_test_labels))

    save = True
    if save:
        model.save('saved_models/holdout-baseline-Adagrad-128')

    return model



def create_trip_model(train_images, train_labels, test_images, test_labels, svc=True):
    
    numerical_train_labels = train_labels.replace(['Amsterdam', 'Barcelona', 'Bucharest', 'Budapest', 'Istanbul', 'London', 'Paris', 'Rome', 'Stockholm', 'Vienna'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    numerical_test_labels = test_labels.replace(['Amsterdam', 'Barcelona', 'Bucharest', 'Budapest', 'Istanbul', 'London', 'Paris', 'Rome', 'Stockholm', 'Vienna'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    model = Sequential()
    model.add(layers.Conv2D(32, (4,4), input_shape=((train_images[0].shape)), activation='relu', padding='same'))
    # model.add(layers.Conv2D(32, (4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(4,4),strides=(4,4)))
    model.add(layers.SpatialDropout2D(0.5))
    # model.add(layers.Conv2D(64, (4,4), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(4,4),strides=(4,4)))
    model.add(layers.SpatialDropout2D(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=None))
    model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

    lr = .001
    optimizer = optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=tfa.losses.TripletSemiHardLoss())
    
    # early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # create tensorboard
    #creating unique name for tensorboard directory
    log_dir = "logs/" + str(f'holdout-tripLoss-opt=Adagrad-withDropout-128')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(train_images, numerical_train_labels, epochs=100, validation_data=(test_images, numerical_test_labels), callbacks=[tensorboard_callback, es])
    # model.fit(train_images, numerical_train_labels, epochs=10, validation_data=(test_images, numerical_test_labels))

    save = True
    if save:
        model.save('saved_models/holdout-tripLoss-Adagrad-128-withDropout')

    if svc:
        train_embeddings = model.predict(train_images)
        test_embeddings = model.predict(test_images)

        # create svm to classify embeddings
        clf = svm.SVC()
        clf.fit(train_embeddings, numerical_train_labels)
        y_pred = clf.predict(test_embeddings)
        print(classification_report(numerical_test_labels, y_pred))

    return model


def main():

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

    # create_base_model(train_images, train_df.City, test_images, test_df.City)
    # create_trip_model(train_images, train_df.City, test_images, test_df.City)

    # change the data to hold 2 cities out and train on that
    # Bucharest has 849, Paris last entry at 1693
    train_df_small = train_df.iloc[1694:, :]
    test_df_small = test_df.iloc[280:, :]
    train_images_small = train_images[1694:]
    test_images_small = test_images[280:]

    train_images_holdout = train_images[:1694]
    train_df_holdout = train_df.iloc[:1694, :]

    # small_base_model = create_base_model(train_images_small, train_df_small.City, test_images_small, test_df_small.City)
    # small_trip_model = create_trip_model(train_images_small, train_df_small.City, test_images_small, test_df_small.City)

    trip_model_holdout = tf.keras.models.load_model('saved_models/holdout-tripLoss-Adagrad-128-withDropout')
    y_pred = trip_model_holdout.predict(train_images_small)
    # y_pred = trip_model_holdout.predict(train_images_holdout)

    # get the baseline model, remove the softmax layer
    # baseline_model_holdout = tf.keras.models.load_model('saved_models/holdout-baseline-Adagrad-128')
    # chopped_baseline_model = Sequential()
    # for layer in baseline_model_holdout.layers[:-1]:
    #     chopped_baseline_model.add(layer)

    # y_pred = chopped_baseline_model.predict(train_images_small)

    true_match = 0
    false_match = 0
    true_different = 0
    false_different = 0
    # for i in range(845):
    #     first = y_pred[i]
    #     second_bad = y_pred[i+1+846]
    #     # second_bad = y_pred[i+1+1692]
    #     # second_bad = y_pred[i+1 + 2538]
    #     second = y_pred[i+1]

    #     # take square of differences and sum them
    #     l2 = np.sum(np.power((first-second),2))
    #     l2_bad = np.sum(np.power((first-second_bad),2))

    #     # if l2 < 0.5:
    #     #     all += l2
    #     # if l2 < 1.242:

    #     split = 0.06
    #     if l2 < split:
    #         same += 1
    #     else:
    #         different += 1

    #     if l2_bad < split:
    #         same_bad += 1
    #     else:
    #         different_bad += 1


    # normalize vector
    y_pred = y_pred / np.max(y_pred)

    # we need to find the best split for both the baseline and the trip loss model
    split = 1.4
    # labels = list(train_df_small.City)
    labels = list(train_df_small.City)
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
        
        if i % 100 == 0:
            print(i)



    # print(f'Average: {all / 845}')
    print(f'true_match: {true_match}')
    print(f'true_different: {true_different}')

    print()

    print(f'false_match: {false_match}')
    print(f'false_different: {false_different}')

    print(f'accuracy: {(true_match+true_different)/(false_match+false_different+true_match+true_different)}')


if __name__ == '__main__':
    main()