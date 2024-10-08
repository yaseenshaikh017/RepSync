from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np

# Create a Blueprint for the upright rows exercise
uprightrows_app = Blueprint('uprightrows_app', __name__)

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

                # Get coordinates for the left shoulder, elbow, and wrist
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate the angle at the shoulder and elbow
                shoulder_angle = calculate_angle(shoulder, elbow, wrist)
                elbow_angle = calculate_angle(shoulder, elbow, [elbow[0], elbow[1] - 0.5])  # Relative to vertical position
                
                # Calculate the vertical distance between wrist and shoulder
                wrist_to_shoulder_height_diff = wrist[1] - shoulder[1]
                
                # Thresholds for accuracy
                min_angle_down = 160  # Arm extended down
                max_angle_up = 70     # Arm bent, elbow flexed upward
                min_height_diff_up = -0.05  # Minimum height difference for wrist to be considered up
                
                # Upright row counter logic
                if shoulder_angle > min_angle_down and wrist_to_shoulder_height_diff > min_height_diff_up:
                    stage = "down"
                if shoulder_angle < max_angle_up and wrist_to_shoulder_height_diff < min_height_diff_up and stage == 'down':
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

@uprightrows_app.route('/uprightrows')
def uprightrows():
    return render_template('model_page.html', model_name='Upright Rows')

@uprightrows_app.route('/video_feed_uprightrows')
def video_feed_uprightrows():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@uprightrows_app.route('/update_data_uprightrows')
def update_data_uprightrows():
    global counter, high_score, stage
    return jsonify(stage=stage, counter=counter, high_score=high_score)

@uprightrows_app.route('/reset_counter_uprightrows', methods=['POST'])
def reset_counter_uprightrows():
    global counter, stage
    counter = 0  # Reset the reps counter only
    stage = None
    return jsonify(success=True)

def reset_uprightrows():
    global counter, stage, high_score
    counter = 0
    stage = None
    high_score = 0