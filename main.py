import pandas as pd
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers

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
    model = Sequential()

    # very small CNN from project 3 of this class

    # print("Building Model")
    # model.add(layers.Conv2D(40, (5, 5), input_shape=((train_images[0].shape)), activation='relu'))
    # model.add(layers.MaxPooling2D(2, 2))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(100, activation='relu'))
    # model.add(layers.Dense(10, activation='softmax'))

    # print("setting params")
    # lr = .001
    # optimizer = optimizers.Adam(learning_rate=lr)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    # VGG16 (minus the first couple convolutional layers since we're starting with a 112x112 instead of 224x224)
    #convolutional layers
    model.add(layers.Conv2D(input_shape=(112, 112, 3),filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    #fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096,activation="relu"))
    model.add(layers.Dense(units=4096,activation="relu"))
    model.add(layers.Dense(units=10, activation="softmax"))
    
    lr = .001
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print('about to fit')
    encoded_train_labels = pd.get_dummies(train_labels)
    encoded_test_labels = pd.get_dummies(test_labels)
    print(encoded_train_labels)
    model.fit(train_images, encoded_train_labels, epochs=10, validation_data=(test_images, encoded_test_labels), )

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

    print(train_images.shape)
    print(train_df.City.shape)
    print(test_images.shape)
    print(test_df.City.shape)

    cv2.imwrite('train.jpg', train_images[0]*255)
    cv2.imwrite('test.jpg', test_images[0]*255)

    create_base_model(train_images, train_df.City, test_images, test_df.City)

if __name__ == '__main__':
    main()