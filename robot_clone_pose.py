import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
#  CONFIGURATION / INITIAL SETUP
# -----------------------------
# Mediapipe Pose objects
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# OpenCV window name
WINDOW_NAME = "👤  You  |  🤖  Robot Clone"

# Define pairs of landmark indices to draw skeleton lines (MediaPipe Pose)
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS  # uses built-in list of connections

# -----------------------------
#  UTILITY FUNCTIONS
# -----------------------------
def normalize_to_pixel(normalized_landmark, frame_width, frame_height):
    """
    Convert a normalized landmark (x, y in [0..1]) to pixel coordinates.
    Returns (int_x, int_y).
    """
    return (
        int(normalized_landmark.x * frame_width),
        int(normalized_landmark.y * frame_height)
    )

def draw_robot_skeleton(canvas, landmarks, color=(200, 200, 200), thickness=2, mirror_x=False):
    """
    Draw a simple stick-figure robot onto `canvas` using the given landmarks.
    - `canvas`: BGR image (NumPy array) where we draw the robot.
    - `landmarks`: List of normalized landmarks from MediaPipe.
    - `color`: BGR tuple for lines/circles.
    - `thickness`: line thickness.
    - `mirror_x`: if True, flip x horizontally around the vertical center of canvas.
    """
    h, w = canvas.shape[:2]

    # First, convert all normalized landmarks to pixel coordinates (with mirror if needed)
    pixel_points = {}
    for idx, lm in enumerate(landmarks):
        px, py = normalize_to_pixel(lm, w, h)
        if mirror_x:
            px = w - px  # flip horizontally
        pixel_points[idx] = (px, py)

    # Draw connections (lines) between joints
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx in pixel_points and end_idx in pixel_points:
            pt1 = pixel_points[start_idx]
            pt2 = pixel_points[end_idx]
            cv2.line(canvas, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    # Draw joints as circles
    for idx, (px, py) in pixel_points.items():
        cv2.circle(canvas, (px, py), radius=4, color=color, thickness=-1, lineType=cv2.LINE_AA)

# -----------------------------
#  MAIN LOOP
# -----------------------------
def main():
    # 1. Open the default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️  Could not open webcam.")
        return

    # 2. Initialize MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to grab frame.")
                break

            # Flip the frame horizontally for a mirror effect on the live camera side
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]

            # 3. Run Pose Detection
            # Convert BGR to RGB before sending to MediaPipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = pose.process(img_rgb)
            img_rgb.flags.writeable = True

            # 4. Prepare a blank canvas for the robot (same size as original)
            robot_canvas = np.zeros_like(frame)  # black background

            # 5. If we have valid landmarks, draw the "robot clone" on the right
            if results.pose_landmarks:
                draw_robot_skeleton(
                    robot_canvas,
                    results.pose_landmarks.landmark,
                    color=(200, 200, 200),
                    thickness=2,
                    mirror_x=True  # make the robot face you
                )

            # 6. Combine left (camera) and right (robot) side by side
            combined = np.hstack((frame, robot_canvas))

            # 7. Show the result
            cv2.imshow(WINDOW_NAME, combined)

            # 8. Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                # Press ESC or 'q' to exit
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
