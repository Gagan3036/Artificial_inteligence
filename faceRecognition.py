import cv2
import numpy as np
import os

# Haar cascade file for face detection
haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Dataset directory
datasets = 'dataset'

print('Training...')

# Initialize lists for images, labels, and a dictionary for names
(images, labels, names, id) = ([], [], {}, 0)

# Desired dimensions for the face images
(width, height) = (130, 100)

# Load images from the dataset directory
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (width, height))
                images.append(img_resized)
                labels.append(int(label))
        id += 1

# Convert lists to numpy arrays
(images, labels) = [np.array(lis) for lis in [images, labels]]
print(images.shape, labels.shape)

# Initialize the LBPH face recognizer
model = cv2.face.LBPHFaceRecognizer_create()
# Uncomment the following line to use FisherFaceRecognizer instead
# model = cv2.face.FisherFaceRecognizer_create()

# Train the model
model.train(images, labels)

# Start video capture
webcam = cv2.VideoCapture(0)
cnt = 0

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if prediction[1] < 800:
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("Unknown.jpg", im)
                cnt = 0
    cv2.imshow('FaceRecognition', im)
    key = cv2.waitKey(10)
    if key == 27:  # Press 'ESC' to exit
        break

webcam.release()
cv2.destroyAllWindows()
