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

def create_base_model(train_images, train_labels, test_images, test_labels, num_cities, use_tensorboard=False):
    encoded_train_labels = pd.get_dummies(train_labels)
    encoded_test_labels = pd.get_dummies(test_labels)
    
    # early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # create tensorboard
    #creating unique name for tensorboard directory
    log_dir = "final_logs/" + str(f'holdout-baseline-opt=Adagrad-128-withDropout')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # custom made CNN
    model = Sequential()
    model.add(layers.Conv2D(32, (4,4), input_shape=((train_images[0].shape)), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(4,4),strides=(4,4)))
    model.add(layers.SpatialDropout2D(0.5))
    model.add(layers.Conv2D(64, (4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(4,4),strides=(4,4)))
    model.add(layers.SpatialDropout2D(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_cities, activation='softmax'))

    lr = .001
    optimizer = optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if use_tensorboard:
        model.fit(train_images, encoded_train_labels, epochs=200, validation_data=(test_images, encoded_test_labels), callbacks=[tensorboard_callback, es])
    else:
        model.fit(train_images, encoded_train_labels, epochs=200, validation_data=(test_images, encoded_test_labels), callbacks=[es])

    save = False
    if save:
        model.save('saved_models_h5/holdout-baseline-Adagrad-128-withDropout.h5')

    return model



def create_trip_model(train_images, train_labels, test_images, test_labels, svc=True, holdout=False, use_tensorboard=False):
    
    if holdout:
        numerical_train_labels = train_labels.replace(['Amsterdam', 'Barcelona', 'Budapest', 'Istanbul', 'London', 'Rome', 'Stockholm', 'Vienna'], [0, 1, 2, 3, 4, 5, 6, 7])
        numerical_test_labels = test_labels.replace(['Amsterdam', 'Barcelona', 'Budapest', 'Istanbul', 'London', 'Rome', 'Stockholm', 'Vienna'], [0, 1, 2, 3, 4, 5, 6, 7])
    else:
        numerical_train_labels = train_labels.replace(['Amsterdam', 'Barcelona', 'Bucharest', 'Budapest', 'Istanbul', 'London', 'Paris', 'Rome', 'Stockholm', 'Vienna'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        numerical_test_labels = test_labels.replace(['Amsterdam', 'Barcelona', 'Bucharest', 'Budapest', 'Istanbul', 'London', 'Paris', 'Rome', 'Stockholm', 'Vienna'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 

    model = Sequential()
    model.add(layers.Conv2D(32, (4,4), input_shape=((train_images[0].shape)), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(4,4),strides=(4,4)))
    model.add(layers.SpatialDropout2D(0.5))
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
    log_dir = "final_logs/" + str(f'holdout-tripLoss-opt=Adagrad-withDropout-128')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if use_tensorboard:
        model.fit(train_images, numerical_train_labels, epochs=200, validation_data=(test_images, numerical_test_labels), callbacks=[tensorboard_callback, es])
    else:
        model.fit(train_images, numerical_train_labels, epochs=200, validation_data=(test_images, numerical_test_labels), callbacks=[es])

    save = False
    if save:
        model.save('saved_models_h5/holdout-tripLoss-Adagrad-128-withDropout.h5')

    if svc:
        train_embeddings = model.predict(train_images)
        test_embeddings = model.predict(test_images)

        # create svm to classify embeddings
        clf = svm.SVC()
        clf.fit(train_embeddings, numerical_train_labels)
        y_pred = clf.predict(test_embeddings)
        print(classification_report(numerical_test_labels, y_pred, digits=4))

    return model


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

    create_new_models = False
    if create_new_models:
        create_base_model(train_images, train_df.City, test_images, test_df.City, 10)
        create_trip_model(train_images, train_df.City, test_images, test_df.City)
    
    # change the data to hold 2 cities out and train on that
    # Bucharest has 849, Paris last entry at 1693
    train_df_small = train_df.iloc[1694:, :]
    test_df_small = test_df.iloc[280:, :]
    train_images_small = train_images[1694:]
    test_images_small = test_images[280:]

    train_images_holdout = train_images[:1694]
    train_df_holdout = train_df.iloc[:1694, :]

    create_new_holdout_models = False
    if create_new_holdout_models:
        small_base_model = create_base_model(train_images_small, train_df_small.City, test_images_small, test_df_small.City, 8)
        small_trip_model = create_trip_model(train_images_small, train_df_small.City, test_images_small, test_df_small.City)

if __name__ == '__main__':
    main()