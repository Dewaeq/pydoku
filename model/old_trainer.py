import tensorflow.keras as keras
from keras.utils import np_utils

# load the mnist dataset directly with tensorflow, amazing!
mnist = keras.datasets.mnist

# x represents the image, y represents the label
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = keras.utils.normalize(X_train, axis=1)
X_test = keras.utils.normalize(X_test, axis=1)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
          kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu',
          kernel_initializer='he_uniform'))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu',
          kernel_initializer='he_uniform'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu',
          kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test),  epochs=50, batch_size=32)
model.save("./model/f.h5")
