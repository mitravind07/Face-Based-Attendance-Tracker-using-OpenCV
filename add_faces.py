#!/usr/bin/env python3
import cv2
import pickle
import numpy as np
import os

# Load the cascade classifier for face detection
face_cascade_path = "data/haarcascade_frontalface_default.xml"
facedetect = cv2.CascadeClassifier(face_cascade_path)

# Fixed size for resized images
RESIZE_WIDTH = 50
RESIZE_HEIGHT = 50


def extract_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    extracted_faces = []
    for x, y, w, h in faces:
        crop_img = frame[y : y + h, x : x + w, :]
        resized_img = cv2.resize(crop_img, (RESIZE_WIDTH, RESIZE_HEIGHT))
        extracted_faces.append(resized_img)
    return extracted_faces


def process_faces_data(faces_data):
    # Convert faces_data to a numpy array and reshape it
    faces_data = np.array(faces_data)
    if len(faces_data.shape) == 4:  # For image capture
        faces_data = faces_data.reshape(
            faces_data.shape[0], RESIZE_WIDTH * RESIZE_HEIGHT * 3
        )
    return faces_data


def upload_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        print("Error: Could not open or read the image file.")
        return []
    faces = extract_faces(img)
    return faces


def live_video_capture():
    faces_data = []
    i = 0
    name = input("Enter Your Name: ")
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        faces = extract_faces(frame)
        for face in faces:
            if len(faces_data) < 100:
                faces_data.append(face)
            i += 1
            cv2.putText(
                frame,
                str(len(faces_data)),
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (50, 50, 255),
                1,
            )
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == ord("q") or len(faces_data) == 100:
            break
    video.release()
    cv2.destroyAllWindows()
    faces_data = process_faces_data(faces_data)
    return faces_data, name


def main():
    choice = input("Choose an option:\n1. Upload image\n2. Live video capture\n")

    if choice == "1":
        file_path = input("Enter the path to the image file: ")
        faces_data = upload_image(file_path)
        faces_data = process_faces_data(faces_data)
        if len(faces_data) == 0:
            exit()
        name = input("Enter Your Name: ")
    elif choice == "2":
        faces_data, name = live_video_capture()
    else:
        print("Invalid choice.")
        exit()

    # Load existing data or create new pickle files
    names_path = "data/names.pkl"
    faces_data_path = "data/faces_data.pkl"

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(names_path):
        names = [name] * len(faces_data)
        with open(names_path, "wb") as f:
            pickle.dump(names, f)
    else:
        with open(names_path, "rb") as f:
            names = pickle.load(f)
        names += [name] * len(faces_data)
        with open(names_path, "wb") as f:
            pickle.dump(names, f)

    if not os.path.exists(faces_data_path):
        with open(faces_data_path, "wb") as f:
            pickle.dump(faces_data, f)
    else:
        with open(faces_data_path, "rb") as f:
            existing_faces_data = pickle.load(f)
        # Convert to numpy array and reshape if necessary
        existing_faces_data = np.array(existing_faces_data)
        if len(existing_faces_data.shape) == 4:
            existing_faces_data = existing_faces_data.reshape(
                existing_faces_data.shape[0], RESIZE_WIDTH * RESIZE_HEIGHT * 3
            )
        faces_data = np.array(faces_data)
        faces_data = faces_data.reshape(
            faces_data.shape[0], RESIZE_WIDTH * RESIZE_HEIGHT * 3
        )
        faces = np.append(existing_faces_data, faces_data, axis=0)
        with open(faces_data_path, "wb") as f:
            pickle.dump(faces, f)


if __name__ == "__main__":
    main()
