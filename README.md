🤖 Pose Robot Clone Visualizer
A fun real-time application using OpenCV and MediaPipe to mirror your webcam pose on the left and a robot skeleton clone on the right — like a digital doppelgänger!

🧠 What It Does
Uses your webcam to detect your body pose live using MediaPipe's Pose model

Displays you on the left and a mirrored robot skeleton on the right

Great for visualizing body landmarks, stick-figure motion tracking, or just having fun!

🎥 Live Preview
diff
Copy
Edit
👤 You (Live Feed)     🤖 Robot Clone (Stick Figure)
+------------------+   +-------------------------+
|  [Webcam Video]  |   |    [Mirrored Skeleton]  |
+------------------+   +-------------------------+
📦 Requirements
Make sure the following packages are installed:

bash
Copy
Edit
pip install opencv-python mediapipe numpy
🚀 How to Run
Clone this repository or save the script.

Run the Python file:

bash
Copy
Edit
python pose_robot_clone.py
Allow access to your webcam.

Press q or ESC to exit.