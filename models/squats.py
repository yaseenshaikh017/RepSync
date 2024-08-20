from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np

# Create a Blueprint for the squat exercise
squats_app = Blueprint('squats_app', __name__)

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
                
                # Get coordinates for the left hip, knee, and ankle
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # Calculate angle
                angle = calculate_angle(hip, knee, ankle)
                
                # Squat counter logic
                if angle > 160:
                    stage = "up"
                if angle < 90 and stage == 'up':
                    stage = "down"
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

@squats_app.route('/squats')
def squats():
    return render_template('model_page.html', model_name='Squats')

@squats_app.route('/video_feed_squats')
def video_feed_squats():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@squats_app.route('/update_data_squats')
def update_data_squats():
    global counter, high_score, stage
    return jsonify(stage=stage, counter=counter, high_score=high_score)

@squats_app.route('/reset_counter_squats', methods=['POST'])
def reset_counter_squats():
    global counter, stage
    counter = 0  # Reset the reps counter only
    stage = None
    return jsonify(success=True)

def reset_squats():
    global counter, stage, high_score
    counter = 0
    stage = None
    high_score = 0