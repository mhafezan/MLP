import os
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# To suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hyperparameters
image_size = 500
batch_size = 128
num_classes = 26  # Number of classes (A-Z)

# Data preprocessing function using ImageDataGenerator to Normalize pixel values
img_dataset_generator = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

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
    shuffle=True)

# To extract the number of samples for each generated dataset
train_samples = dataset_train.samples
val_samples = dataset_val.samples

# Model definition (same MLP architecture as MNIST example)
model = Sequential()
model.add(Input(shape=(image_size*image_size*3,))) # Flattened input layer for 500x500 RGB images
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
    epochs=10,
    validation_data=dataset_val,
    validation_steps=val_samples // batch_size
)

# Evaluate the model on the validation set
test_loss, test_acc = model.evaluate(dataset_val, steps=val_samples // batch_size)
print(f"Validation Accuracy: {test_acc * 100:.2f}%")