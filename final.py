import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.degrees(radians)
    if angle < 0:
        angle += 360
    return angle

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark

        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]]
        right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0]]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]]

        left_hunchback_angle = calculate_angle(left_ear, left_shoulder, left_shoulder)
        right_hunchback_angle = calculate_angle(right_ear, right_shoulder, right_shoulder)
        left_lordosis_angle = calculate_angle(left_shoulder, left_hip, [left_hip[0], left_hip[1] + 1])
        right_lordosis_angle = calculate_angle(right_shoulder, right_hip, [right_hip[0], right_hip[1] + 1])

        cv2.putText(frame, f'L_Hunch: {int(left_hunchback_angle)}', (int(left_shoulder[0]), int(left_shoulder[1] - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'R_Hunch: {int(right_hunchback_angle)}', (int(right_shoulder[0]), int(right_shoulder[1] - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'L_Lord: {int(left_lordosis_angle)}', (int(left_hip[0]), int(left_hip[1] - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'R_Lord: {int(right_lordosis_angle)}', (int(right_hip[0]), int(right_hip[1] - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if left_hunchback_angle > 100 or right_hunchback_angle > 130:
            cv2.putText(frame, 'Kyphosis Detected', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Recommended Yoga Pose: Cobra Pose', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif left_lordosis_angle > 190 or right_lordosis_angle > 190:
            cv2.putText(frame, 'Lordosis Detected', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Recommended Yoga Pose: Child\'s Pose', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Good Posture', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Pose', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()