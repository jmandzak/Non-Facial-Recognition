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
    
    print("Building Model")
    model = Sequential()
    model.add(layers.Conv2D(40, (5, 5), input_shape=((train_images[0].shape)), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    print("setting params")
    lr = .001
    batch_size = 100
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print('about to fit')
    model.fit(train_images, train_labels, batch_size=batch_size, epochs=50, validation_data=(test_images, test_labels), )

def main():
    train_df = make_df(train)
    test_df = make_df(test)

    # create master list
    train_images = np.zeros((8477, 224, 224, 3))

    # read in images
    for i, filename in enumerate(train_df.File):
        image = cv2.imread(filename)
        # normalize and resize the image
        image = image / 255
        image = cv2.resize(image, (224, 224))

        train_images[i] = image


    print('yo')
    test_images = np.zeros((1400, 224, 224, 3))
    for i, filename in enumerate(test_df.File):
        image = cv2.imread(filename)
        # normalize and resize the image
        image = image / 255
        image = cv2.resize(image, (224, 224))

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