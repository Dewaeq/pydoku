""" import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop


def read_data_files():
    PATH = "c:/src/projects/python/sudoku/train_data/digits"

    x_data = []
    y_data = []

    for label in range(0, 10):
        files = os.listdir(os.path.join(PATH, str(label)))

        for file in files:
            img = cv2.imread(os.path.join(PATH, str(label), file))
            img = cv2.resize(img, (32, 32))

            x_data.append(img)
            y_data.append(label)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


def process_data(x_data, y_data):
    # Split data in training and testing sets
    # use 10 % as testing data
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.10)
    # use 18 % as validation data (0.9 * 0.2)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.20)

    print("Training Set Shape: ", x_train.shape)
    print("Validation Set Shape: ", x_valid.shape)
    print("Test Set Shape: ", x_test.shape)

    def preprocess(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        result = eq / 255

        return result

    x_train = map(preprocess, x_train)
    x_test = map(preprocess, x_test)
    x_valid = map(preprocess, x_valid)

    x_train = np.array(list(x_train))
    x_test = np.array(list(x_test))
    x_valid = np.array(list(x_valid))

    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_valid = x_valid.reshape(
        x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1)

    # Add some randomness to the data
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                 zoom_range=0.2, shear_range=0.1, rotation_range=10)
    datagen.fit(x_train)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    y_valid = to_categorical(y_valid, 10)

    return (x_train, x_test, x_valid), (y_train, y_test, y_valid), datagen


def build_model():
    model = keras.models.Sequential()

    model.add(Conv2D(60, (5, 5), padding="same",
              activation="relu", input_shape=(32, 32, 1)))
    model.add(Conv2D(60, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(30, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(30, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    return model


x_data, y_data = read_data_files()
print("Data size: ", len(x_data))

(x_train, x_test, x_valid), (y_train, y_test,
                             y_valid), datagen = process_data(x_data, y_data)

model = build_model()
model.summary()

# Compiling the model
optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                    epochs=30, validation_data=(x_valid, y_valid),
                    verbose=1, steps_per_epoch=200)

# Testing the model on the test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score: ', score[0])
print('Test Accuracy: ', score[1])

model.save("model.h5")
 """