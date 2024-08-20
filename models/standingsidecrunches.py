from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np

# Create a Blueprint for the standing side crunches exercise
standingsidecrunches_app = Blueprint('standingsidecrunches_app', __name__)

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

                # Get coordinates for the left elbow, left hip, and left knee
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                # Calculate the vertical distance between the elbow and knee
                elbow_to_knee_distance = calculate_distance(left_elbow, left_knee)

                # Calculate the vertical distance between the elbow and hip
                elbow_to_hip_distance = calculate_distance(left_elbow, left_hip)

                # Thresholds for accuracy
                min_elbow_to_knee_distance = 0.1  # Minimum distance to consider as a crunch
                max_elbow_to_hip_distance = 0.2   # Maximum distance when elbow is near the hip

                # Side crunch counter logic
                if elbow_to_knee_distance < min_elbow_to_knee_distance:
                    stage = "crunch"
                if elbow_to_hip_distance < max_elbow_to_hip_distance and stage == "crunch":
                    stage = "reset"
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

@standingsidecrunches_app.route('/standingsidecrunches')
def standingsidecrunches():
    return render_template('model_page.html', model_name='Standing Side Crunches')

@standingsidecrunches_app.route('/video_feed_standingsidecrunches')
def video_feed_standingsidecrunches():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@standingsidecrunches_app.route('/update_data_standingsidecrunches')
def update_data_standingsidecrunches():
    global counter, high_score, stage
    return jsonify(stage=stage, counter=counter, high_score=high_score)

@standingsidecrunches_app.route('/reset_counter_standingsidecrunches', methods=['POST'])
def reset_counter_standingsidecrunches():
    global counter, stage
    counter = 0  # Reset the reps counter only
    stage = None
    return jsonify(success=True)

def reset_standingsidecrunches():
    global counter, stage, high_score
    counter = 0
    stage = None
    high_score = 0