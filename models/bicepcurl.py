from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np

# Create a Blueprint for the bicep curl
bicepcurl_app = Blueprint('bicepcurl_app', __name__)

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
    scaling_factor = 1.5  # Adjusted for more strictness

    # Dynamic thresholds based on distances
    max_angle = 180 - (shoulder_elbow_dist + elbow_wrist_dist) * scaling_factor
    min_angle = 30 + (shoulder_elbow_dist + elbow_wrist_dist) * scaling_factor

    return max_angle, min_angle

def gen_frames():
    global counter, stage, high_score
    cap = cv2.VideoCapture(0)  # Open the camera
    previous_angle = None
    hold_start_time = None  # To track the hold time at the "up" position
    frames_in_position = 0  # To ensure the user holds the position

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

                # Calculate angles
                angle = calculate_angle(shoulder, elbow, wrist)

                # Calculate dynamic thresholds
                max_angle, min_angle = dynamic_thresholds(shoulder, elbow, wrist)

                # Smooth angle using exponential moving average (optional)
                if previous_angle is not None:
                    smoothed_angle = 0.8 * previous_angle + 0.2 * angle
                else:
                    smoothed_angle = angle

                previous_angle = smoothed_angle

                # Calculate the horizontal distance between the shoulder and wrist to ensure proper arm alignment
                shoulder_wrist_dist = np.abs(shoulder[0] - wrist[0])

                # Check for arm alignment, velocity, and angle thresholds
                if shoulder_wrist_dist > 0.1 and smoothed_angle > max_angle:
                    stage = "down"
                    hold_start_time = None  # Reset hold time when the arm is fully extended
                    frames_in_position = 0  # Reset frame counter

                if shoulder_wrist_dist > 0.1 and min_angle < smoothed_angle < 30 and stage == 'down':
                    stage = "up"
                    if hold_start_time is None:
                        hold_start_time = cv2.getTickCount()

                    # Check if the user held the "up" position for at least 0.5 seconds and maintain the position for a few frames
                    hold_time = (cv2.getTickCount() - hold_start_time) / cv2.getTickFrequency()
                    frames_in_position += 1

                    if hold_time > 0.5 and frames_in_position > 5:  # Ensuring the user holds the position for at least 5 frames
                        counter += 1
                        stage = "counted"  # Change stage to a new value to prevent multiple increments
                        frames_in_position = 0  # Reset frame counter
                        if counter > high_score:
                            high_score = counter

                # Reset the stage back to "down" only when the arm is fully extended again
                if shoulder_wrist_dist > 0.1 and smoothed_angle > max_angle and stage == "counted":
                    stage = "down"

            except Exception as e:
                print(f"Error: {e}")

        # Render the video feed with the annotations
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@bicepcurl_app.route('/bicepcurl')
def bicepcurl():
    return render_template('model_page.html', model_name='Bicep Curl')

@bicepcurl_app.route('/video_feed_bicepcurl')
def video_feed_bicepcurl():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bicepcurl_app.route('/update_data_bicepcurl')
def update_data_bicepcurl():
    global counter, high_score, stage
    return jsonify(stage=stage, counter=counter, high_score=high_score)

@bicepcurl_app.route('/reset_counter_bicepcurl', methods=['POST'])
def reset_counter_bicepcurl():
    global counter, stage
    counter = 0  # Reset the reps counter only
    stage = None
    return jsonify(success=True)

def reset_bicepcurl():
    global counter, stage, high_score
    counter = 0
    stage = None
    high_score = 0
