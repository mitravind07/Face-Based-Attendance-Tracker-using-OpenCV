# 👤 Face Recognition Attendance System

This project implements a real-time **Face Recognition Attendance System** using Python, OpenCV, and scikit-learn. It captures faces from a live video stream, recognizes them using trained models, and automatically logs attendance.

---

## ✨ Features

- 🔍 Real-time face detection using Haar cascades (OpenCV)
- 🧠 Face recognition using machine learning (scikit-learn)
- 📸 Add new faces and update training data easily
- 🗓️ Auto-generate attendance logs with date-wise records
- 🖥️ Simple keyboard control for capture and exit

---

## 🧰 Requirements

Make sure Python 3.x is installed, then install the required libraries:

```bash
pip install opencv-python numpy pandas scikit-learn
```

Also ensure the file `haarcascade_frontalface_default.xml` is present in the `data/` directory.

---

## 📂 Folder Structure

```
Face-Recognition-main/
├── face_recognition_project/
│   ├── add_faces.py                  # Capture new face data
│   ├── test.py                       # Real-time recognition and attendance
│   ├── data/
│   │   ├── faces_data.pkl            # Serialized face encodings
│   │   ├── names.pkl                 # Serialized names
│   │   └── haarcascade_frontalface_default.xml
│   ├── Attendance/
│   │   └── Attendance_<date>.csv     # Daily attendance logs
└── README.md
```

---

## 🚀 How to Use

1. **Add Faces**  
   Run the following command and follow on-screen prompts to capture face data:
   ```bash
   python add_faces.py
   ```

2. **Start Recognition and Track Attendance**
   ```bash
   python test.py
   ```

   | Key | Function                |
   |-----|-------------------------|
   | `p` | Mark attendance         |
   | `q` | Quit the program        |

---

## 📝 Output

- New face data is stored using `pickle`.
- Attendance is logged into a CSV file in the `Attendance/` folder with the current date.

---

## 📬 Contact

For questions or feedback, feel free to reach out:

- **Email**: [mitravind07@gmail.com](mailto:mitravind07@gmail.com)  
- **GitHub**: [mitravind07](https://github.com/mitravind07)
