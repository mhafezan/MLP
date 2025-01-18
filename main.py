import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the input data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Flatten the images
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = Sequential()
# Add input layer and first hidden layer
model.add(Dense(512, input_shape=(784,), activation='relu'))
# Add second hidden layer
model.add(Dense(256, activation='relu'))
# Add output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Summarize the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save('mnist_mlp_model.h5')

# Load the Model to use for Prediction
loaded_model = tf.keras.models.load_model('mnist_mlp_model.h5')

# Predict on a test sample (Pick a test sample)
sample = X_test[0].reshape(1, -1)

# Predict the class
prediction = loaded_model.predict(sample)
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted class: {predicted_class}")

# Display the test sample
plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted class: {predicted_class[0]}")
plt.show()