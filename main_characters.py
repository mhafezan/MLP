# To suppress TensorFlow logs
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Hyperparameters
image_size = 500
batch_size = 128
epochs = 10
num_classes = 26  # Number of classes (A-Z)

# Data preprocessing using ImageDataGenerator to Normalize pixel values to a range of [0,1]
img_dataset_generator = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# Dataset Folder Path
dataset_path = "./english_characters_dataset"

dataset_train = img_dataset_generator.flow_from_directory(
    dataset_path,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True)

dataset_val = img_dataset_generator.flow_from_directory(
    dataset_path,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False)

# To extract the number of samples for each generated dataset
train_samples = dataset_train.samples
val_samples = dataset_val.samples

# Model definition (same MLP architecture as MNIST example)
model = Sequential()
model.add(Input(shape=(image_size, image_size, 3)))  # Input layer for image shape (500, 500, 3)
model.add(Flatten())                   # Flatten the input images
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
optimizer = 'adam'
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    dataset_train,
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    validation_data=dataset_val,
    validation_steps=val_samples // batch_size)

# Evaluate the model on the validation set
test_loss, test_acc = model.evaluate(dataset_val, steps=val_samples//batch_size)
print(f"Validation Accuracy: {test_acc * 100:.2f}%")

sys.exit(0)