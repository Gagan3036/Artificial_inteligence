from numpy import loadtxt
from keras.models import model_from_json
import numpy as np

# Load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

# Load the model architecture from JSON file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load the model weights
model.load_weights("model.weights.h5")  # Ensure this matches the filename used in the training script
print("Loaded model from disk")

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Make predictions
predictions = (model.predict(x) > 0.5).astype("int32")  # Use thresholding for binary classification

# Print predictions for samples 5 to 9
for i in range(5, 10):
    print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
