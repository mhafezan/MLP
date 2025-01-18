# To suppress TensorFlow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Important imports
import sys
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# To load the MNIST dataset (Trainset size: 60000, Testset size: 10000, image size: 28*28 )
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# To normalize the input data
"""Pixel values are initially ranged from 0 to 255.
Normalization shifts the pixel values to a range of [0,1]"""
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# To flatten the images before feeding into the first Dense layer
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# To represent categorical data (It converts integer labels (0, 1, ..., 9) into one-hot encoded vectors)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# To create an instance of the Sequential model. It allows for building a linear stack of layers
model = Sequential()
# To define layers in order
model.add(Input(shape=(784,)))             # To add the input layer
model.add(Dense(512, activation='relu'))   # To add the first hidden layer
model.add(Dense(256, activation='relu'))   # To add second hidden layer
model.add(Dense(10, activation='softmax')) # To add output layer

# To define the model configuring Optimizer, Loss function, and Performance Metric 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# To train the model on trainset specifying Epochs, Batch Size, and Validation Percentage (20% here)
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# To evaluate the trained model on testset
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}\n")

# To save the model
model.save('./weights/mnist_mlp_model.keras')

# To load the pre-trained model again to use for Prediction
pretrained_model = tf.keras.models.load_model('./weights/mnist_mlp_model.keras')

# To predict on a specific test sample
"""As the first layer is a FC/Dense layer, we should pass a flatten image to it"""
sample = x_test[0].reshape(1, -1)

# To predict the class for the sample
prediction = pretrained_model.predict(sample)
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted class is: {predicted_class}\n")

# To display the real sample
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted class: {predicted_class[0]}")
plt.show()

sys.exit(0)