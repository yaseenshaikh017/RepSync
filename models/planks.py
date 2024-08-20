from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import time

# Create a Blueprint for the plank exercise
planks_app = Blueprint('planks_app', __name__)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Global variables
plank_start_time = None
plank_duration = 0
plank_active = False
plank_threshold_angle = 160  # Minimum angle to consider the body straight

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
    global plank_start_time, plank_duration, plank_active
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
                
                # Get coordinates for the shoulders, hips, and ankles
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                # Calculate angles to ensure the body is straight
                left_body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
                right_body_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
                
                # Check if both sides of the body are straight enough to be considered in a plank position
                if left_body_angle > plank_threshold_angle and right_body_angle > plank_threshold_angle:
                    if not plank_active:
                        plank_start_time = time.time()
                        plank_active = True
                    plank_duration = time.time() - plank_start_time
                else:
                    plank_active = False
                    plank_duration = 0

            except Exception as e:
                print(f"Error: {e}")

        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@planks_app.route('/planks')
def planks():
    return render_template('model_page.html', model_name='Planks')

@planks_app.route('/video_feed_planks')
def video_feed_planks():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@planks_app.route('/update_data_planks')
def update_data_planks():
    global plank_duration
    return jsonify(plank_duration=plank_duration)

@planks_app.route('/reset_counter_planks', methods=['POST'])
def reset_counter_planks():
    global plank_duration, plank_active
    plank_duration = 0  # Reset the plank duration
    plank_active = False
    return jsonify(success=True)

def reset_planks():
    global plank_duration, plank_active
    plank_duration = 0
    plank_active = False
