from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np

# Create a Blueprint for the push-up exercise
pushups_app = Blueprint('pushups_app', __name__)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Global variables
counter = 0
stage = None
high_score = 0

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

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

                # Get coordinates for specific landmarks
                shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                elbow_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                
                # Push-up logic based on landmark positions
                if elbow_y > shoulder_y and elbow_y > wrist_y and hip_y > knee_y:
                    stage = "down"
                if elbow_y < shoulder_y and elbow_y < wrist_y and hip_y < knee_y and stage == "down":
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

@pushups_app.route('/pushups')
def pushups():
    return render_template('model_page.html', model_name='Push-Ups')

@pushups_app.route('/video_feed_pushups')
def video_feed_pushups():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@pushups_app.route('/update_data_pushups')
def update_data_pushups():
    global counter, high_score, stage
    return jsonify(stage=stage, counter=counter, high_score=high_score)

@pushups_app.route('/reset_counter_pushups', methods=['POST'])
def reset_counter_pushups():
    global counter, stage
    counter = 0  # Reset the reps counter only
    stage = None
    return jsonify(success=True)

def reset_pushups():
    global counter, stage, high_score
    counter = 0
    stage = None
    high_score = 0
