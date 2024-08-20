from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np

# Create a Blueprint for the calf raises
calfraises_app = Blueprint('calfraises_app', __name__)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Global variables
counter = 0 
stage = None
high_score = 0

def calculate_distance(a, b):
    a = np.array(a)  # First point
    b = np.array(b)  # Second point
    distance = np.linalg.norm(a - b)
    return distance

def gen_frames():
    global counter, stage, high_score
    cap = cv2.VideoCapture(0)  # Open the camera

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (640, 480))  # Resize the frame for display
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                                      mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for the left ankle and left heel (using landmarks for foot positions)
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

                # Calculate the vertical distance between heel and ankle
                heel_to_ankle_height_diff = ankle[1] - heel[1]

                # Thresholds for accuracy
                min_height_diff_up = 0.05  # Minimum height difference for calf raise up
                max_height_diff_down = -0.02  # Threshold for calf raise down

                # Calf raise counter logic
                if heel_to_ankle_height_diff < max_height_diff_down:
                    stage = "down"
                if heel_to_ankle_height_diff > min_height_diff_up and stage == 'down':
                    stage = "up"
                    counter += 1
                    print(f"Reps: {counter}")
                    if counter > high_score:
                        high_score = counter

            except Exception as e:
                print(f"Error: {e}")

        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@calfraises_app.route('/calfraises')
def calfraises():
    return render_template('model_page.html', model_name='Calf Raises')

@calfraises_app.route('/video_feed_calfraises')
def video_feed_calfraises():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@calfraises_app.route('/update_data_calfraises')
def update_data_calfraises():
    global counter, high_score, stage
    return jsonify(stage=stage, counter=counter, high_score=high_score)

@calfraises_app.route('/reset_counter_calfraises', methods=['POST'])
def reset_counter_calfraises():
    global counter, stage
    counter = 0  # Reset the reps counter only
    stage = None
    return jsonify(success=True)

def reset_calfraises():
    global counter, stage, high_score
    counter = 0
    stage = None
    high_score = 0
