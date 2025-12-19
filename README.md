## Aim:
    To analyze student attention in a classroom video by detecting face landmarks and estimating head orientation using MediaPipe Face Mesh and OpenCV.

## Objective:

Detect multiple student faces from a classroom video.

Estimate head yaw angle using facial landmarks.

Classify student attention as Attentive or Not Attentive based on head movement.

Generate an attention report for each detected student.

## Software Requirements

Python 3.x

OpenCV (cv2)

MediaPipe

NumPy

## Hardware Requirements

System with minimum 4 GB RAM

Camera-recorded classroom video (MP4 format)

CPU/GPU capable of real-time video processing

## Algorithm / Methodology
Load the classroom video using OpenCV.
Initialize MediaPipe Face Mesh for multi-face detection.
Process each video frame:
Convert frame from BGR to RGB.
Detect facial landmarks.
For each detected face:
Extract left eye, right eye, and nose landmarks.
Compute head yaw using the horizontal difference between eye landmarks.
Classify head orientation:
Left → yaw < −threshold
Right → yaw > threshold
Center → otherwise
Accumulate frame-wise attention statistics.
Calculate distraction percentage.
Classify students as:
Attentive if distraction < 50%
Not Attentive otherwise
System Configuration Parameters
Parameter	Value
Yaw Threshold	15 degrees
Distraction Limit	50%
Maximum Faces	10
Detection Confidence	0.5
Tracking Confidence	0.5

## Input

Classroom video file (classroom.mp4)

## Output

Console-based Student Attention Report displaying:

Left head turns

Right head turns

Center-facing frames

Total frames

Distraction percentage

Attention status

## Progarm:
```
import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_PATH = "classroom.mp4"
YAW_THRESHOLD = 15        # degrees
DISTRACTION_LIMIT = 50   # percent

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# DATA STRUCTURE
# -----------------------------
students = {}

# -----------------------------
# VIDEO LOAD
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    # DEBUG: FACE COUNT
    print("Faces detected:", len(result.multi_face_landmarks) if result.multi_face_landmarks else 0)

    if result.multi_face_landmarks:
        for idx, face_landmarks in enumerate(result.multi_face_landmarks):

            if idx not in students:
                students[idx] = {
                    "left": 0,
                    "right": 0,
                    "center": 0,
                    "total": 0
                }

            nose = face_landmarks.landmark[1]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            yaw = (right_eye.x - left_eye.x) * 100

            if yaw > YAW_THRESHOLD:
                students[idx]["right"] += 1
            elif yaw < -YAW_THRESHOLD:
                students[idx]["left"] += 1
            else:
                students[idx]["center"] += 1

            students[idx]["total"] += 1

cap.release()

# -----------------------------
# FINAL REPORT
# -----------------------------
print("\n----- STUDENT ATTENTION REPORT -----\n")

for student_id, data in students.items():
    distracted = data["left"] + data["right"]
    total = data["total"]

    distraction_percent = (distracted / total) * 100 if total > 0 else 0
    status = "ATTENTIVE" if distraction_percent < DISTRACTION_LIMIT else "NOT ATTENTIVE"

    print(f"Student {student_id}:")
    print(f"  Left Turns    : {data['left']}")
    print(f"  Right Turns   : {data['right']}")
    print(f"  Center Frames : {data['center']}")
    print(f"  Total Frames  : {total}")
    print(f"  Distraction   : {distraction_percent:.2f}%")
    print(f"  Status        : {status}\n")
```
## Result:
This experiment demonstrates a practical approach to monitoring student attention using computer vision techniques. MediaPipe Face Mesh provides accurate facial landmark detection, enabling effective head pose estimation for classroom analytics.
