#!/usr/bin/env python3
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import subprocess

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

# Load data from pickle files
with open("data/names.pkl", "rb") as w:
    LABELS = pickle.load(w)
    print("Number of labels loaded:", len(LABELS))

with open("data/faces_data.pkl", "rb") as f:
    FACES = np.array(pickle.load(f))

print("Original Shape of Faces matrix:", FACES.shape)

# Flatten the images
num_samples, total_features = FACES.shape
print("Number of Samples:", num_samples)
print("Total Features:", total_features)
FACES = FACES.reshape(num_samples, total_features)
print("Shape of Faces matrix:", FACES.shape)

# Initialize KNN Classifier and load model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(FACES, LABELS)

# Debugging: Print out the lengths of LABELS and FACES
print("Length of LABELS:", len(LABELS))
print("Length of FACES:", len(FACES))

# CSV column names
COL_NAMES = ["Name", "Time"]

while True:
    ret, frame = video.read()
    if not ret:
        break  # Exit the loop if there's an issue reading the frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        crop_img = frame[y : y + h, x : x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        # Get timestamp
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        # Draw rectangles and write text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(
            frame,
            str(output[0]),
            (x, y - 15),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
            1,
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

        attendance = [str(output[0]), str(timestamp)]

    # Display frame
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)

    # Take attendance
    if k == ord("p"):
        # Use subprocess to invoke text-to-speech
        subprocess.run(["say", "Attendance Taken.."])
        time.sleep(5)

        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
        break

    # Quit
    if k == ord("q"):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
