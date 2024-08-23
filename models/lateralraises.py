from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np

# Create a Blueprint for the lateral raise exercise
lateralraise_app = Blueprint('lateralraise_app', __name__)

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

def dynamic_thresholds(shoulder, elbow, wrist):
    # Calculate distances
    shoulder_elbow_dist = np.linalg.norm(np.array(shoulder) - np.array(elbow))
    elbow_wrist_dist = np.linalg.norm(np.array(elbow) - np.array(wrist))

    # Set scaling factor for thresholds
    scaling_factor = 1.2  # Adjust this based on user testing

    # Dynamic thresholds based on distances
    max_angle_up = 150 + (shoulder_elbow_dist + elbow_wrist_dist) * scaling_factor
    min_angle_down = 70 - (shoulder_elbow_dist + elbow_wrist_dist) * scaling_factor

    return max_angle_up, min_angle_down

def gen_frames():
    global counter, stage, high_score
    cap = cv2.VideoCapture(0)  # Open the camera
    previous_angle = None
    frames_in_position = 0  # To ensure the user holds the position
    min_frames_in_position = 5  # Require the arm to hold position for 5 frames

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

                # Calculate the angle at the shoulder
                shoulder_angle = calculate_angle(shoulder, elbow, wrist)

                # Calculate dynamic thresholds
                max_angle_up, min_angle_down = dynamic_thresholds(shoulder, elbow, wrist)

                # Smooth angle using exponential moving average (optional)
                if previous_angle is not None:
                    smoothed_angle = 0.8 * previous_angle + 0.2 * shoulder_angle
                else:
                    smoothed_angle = shoulder_angle

                previous_angle = smoothed_angle

                # Lateral raise counter logic with dynamic thresholds and hold time
                if smoothed_angle < min_angle_down:
                    stage = "down"
                    frames_in_position = 0  # Reset frame counter

                if smoothed_angle > max_angle_up and stage == 'down':
                    frames_in_position += 1
                    if frames_in_position >= min_frames_in_position:  # Require the position to be held
                        stage = "up"
                        counter += 1
                        frames_in_position = 0  # Reset frame counter
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

@lateralraise_app.route('/lateralraise')
def lateralraise():
    return render_template('model_page.html', model_name='Lateral Raise')

@lateralraise_app.route('/video_feed_lateralraise')
def video_feed_lateralraise():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@lateralraise_app.route('/update_data_lateralraise')
def update_data_lateralraise():
    global counter, high_score, stage
    return jsonify(stage=stage, counter=counter, high_score=high_score)

@lateralraise_app.route('/reset_counter_lateralraise', methods=['POST'])
def reset_counter_lateralraise():
    global counter, stage
    counter = 0  # Reset the reps counter only
    stage = None
    return jsonify(success=True)

def reset_lateralraise():
    global counter, stage, high_score
    counter = 0
    stage = None
    high_score = 0
