import pandas as pd
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
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

    # very small CNN from project 3 of this class

    # print("Building Model")
    # model = Sequential()
    # model.add(layers.Conv2D(40, (5, 5), input_shape=((train_images[0].shape)), activation='relu'))
    # model.add(layers.MaxPooling2D(2, 2))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(100, activation='relu'))
    # model.add(layers.Dense(10, activation='softmax'))


    # custom made CNN
    # model = Sequential()
    # model.add(layers.Conv2D(32, (4,4), input_shape=((train_images[0].shape)), activation='relu', padding='same'))
    # # model.add(layers.Conv2D(32, (4,4), activation='relu', padding='same'))
    # model.add(layers.MaxPool2D(pool_size=(4,4),strides=(4,4)))
    # # model.add(layers.Conv2D(64, (4,4), activation='relu', padding='same'))
    # model.add(layers.Conv2D(64, (4,4), activation='relu', padding='same'))
    # model.add(layers.MaxPool2D(pool_size=(4,4),strides=(4,4)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(100, activation='relu'))
    # model.add(layers.Dense(10, activation='softmax'))


    # with triplet loss
    model = Sequential()
    model.add(layers.Conv2D(32, (4,4), input_shape=((train_images[0].shape)), activation='relu', padding='same'))
    # model.add(layers.Conv2D(32, (4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(4,4),strides=(4,4)))
    # model.add(layers.Conv2D(64, (4,4), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (4,4), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(4,4),strides=(4,4)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation=None))
    model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))


    # VGG16 (minus the first couple convolutional layers since we're starting with a 112x112 instead of 224x224)
    #convolutional layers
    # model = Sequential()
    # model.add(layers.Conv2D(input_shape=(112, 112, 3),filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    #fully connected layers
    # model.add(layers.Flatten())
    # model.add(layers.Dense(units=4096,activation="relu"))
    # model.add(layers.Dense(units=4096,activation="relu"))
    # model.add(layers.Dense(units=10, activation="softmax"))

    # mobile net
    # model = MobileNetV3Small(input_shape=(112,112,3), classes=10, weights=None)
    
    lr = .001
    optimizer = optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=tfa.losses.TripletSemiHardLoss())
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # print(model.predict(test_images))
    # print('\n\n')
    # print(model.predict(train_images))

    print('about to fit')
    encoded_train_labels = pd.get_dummies(train_labels)
    encoded_test_labels = pd.get_dummies(test_labels)

    # print(encoded_train_labels)
    # print(encoded_train_labels.shape)

    # using integer representation instead of dummies for the triplet semi hard loss model
    numerical_train_labels = train_labels.replace(['Amsterdam', 'Barcelona', 'Bucharest', 'Budapest', 'Istanbul', 'London', 'Paris', 'Rome', 'Stockholm', 'Vienna'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    numerical_test_labels = test_labels.replace(['Amsterdam', 'Barcelona', 'Bucharest', 'Budapest', 'Istanbul', 'London', 'Paris', 'Rome', 'Stockholm', 'Vienna'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(numerical_train_labels)
    # print(numerical_train_labels.shape)

    # model.fit(train_images, encoded_train_labels, epochs=100, validation_data=(test_images, encoded_test_labels), callbacks=[tensorboard_callback])
    # model.fit(train_images, encoded_train_labels, epochs=100, validation_data=(test_images, encoded_test_labels))

    # for training the triplet loss model
    # create tensorboard
    #creating unique name for tensorboard directory
    log_dir = "logs/" + str(f'tripLoss-opt=SGD')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(train_images, numerical_train_labels, epochs=5, validation_data=(test_images, numerical_test_labels), callbacks=tensorboard_callback)

    train_embeddings = model.predict(train_images)
    test_embeddings = model.predict(test_images)

    # create svm to classify embeddings
    clf = svm.SVC()
    clf.fit(train_embeddings, numerical_train_labels)
    y_pred = clf.predict(test_embeddings)
    print(classification_report(numerical_test_labels, y_pred))

    # I'm not even sure if this next model should exist
    # new_model = Sequential()
    # new_model.add(layers.Dense(10, input_dim=256, activation='softmax'))
    # new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # new_model.fit(train_embeddings, encoded_train_labels, epochs=5)

def main():
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


    print('yo')
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

    # print(train_images.shape)
    # print(train_df.City.shape)
    # print(test_images.shape)
    # print(test_df.City.shape)

    cv2.imwrite('train.jpg', train_images[0]*255)
    cv2.imwrite('test.jpg', test_images[0]*255)

    create_base_model(train_images, train_df.City, test_images, test_df.City)

if __name__ == '__main__':
    main()