import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import math
from PIL import Image

# Initialize Streamlit
st.title("Hand Tracking with Streamlit Camera Feed 📷🖐")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.8
)

# Define colors for overlay
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# Sidebar layout for buttons
st.sidebar.title("Controls")
run = st.sidebar.checkbox("Start Camera", value=True)

# Streamlit camera input
frame_window = st.empty()
camera_feed = st.camera_input("Take a picture or use live feed")

def process_frame(image):
    """Process frame for hand tracking using OpenCV and MediaPipe."""
    # Convert to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process with MediaPipe
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            h, w, _ = image.shape
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Draw hand landmarks
            cv2.circle(image, (ix, iy), 10, GREEN, cv2.FILLED)
            cv2.circle(image, (tx, ty), 10, GREEN, cv2.FILLED)
            cv2.line(image, (ix, iy), (tx, ty), GREEN, 5)

    return image

# Main loop for processing the camera feed
if run and camera_feed is not None:
    # Convert Streamlit image to OpenCV
    image = Image.open(camera_feed)
    processed_image = process_frame(image)

    # Convert back to RGB for Streamlit display
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Display processed frame in Streamlit
    frame_window.image(processed_image, use_column_width=True)
