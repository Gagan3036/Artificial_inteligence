from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Define the model
model = Sequential()

# Convolutional layer with 32 filters, each of size 3x3
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening layer
model.add(Flatten())

# Fully connected layer with 128 neurons
model.add(Dense(units=128, activation='relu'))

# Output layer with 1 neuron (binary classification)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image data augmentation for training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Image data augmentation for validation set
val_datagen = ImageDataGenerator(rescale=1./255)

# Load and augment training set
training_set = train_datagen.flow_from_directory('Dataset/train',
                                                 target_size=(64, 64),
                                                 batch_size=8,
                                                 class_mode='binary')

# Load and augment validation set
val_set = val_datagen.flow_from_directory('Dataset/val',
                                          target_size=(64, 64),
                                          batch_size=8,
                                          class_mode='binary')

# Train the model
model.fit_generator(training_set,
                    steps_per_epoch=10,
                    epochs=50,
                    validation_data=val_set,
                    validation_steps=2)

# Save the model architecture to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("model.h5")
print("Saved model to disk")
