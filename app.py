import streamlit as st
import cv2
import time

# Initialize session state
if 'run' not in st.session_state:
    st.session_state.run = False

# Toggle webcam state
def toggle_webcam():
    st.session_state.run = not st.session_state.run

# Webcam control buttons
col1, col2 = st.columns(2)
with col1:
    start_btn = st.button("Start Webcam", on_click=toggle_webcam)
with col2:
    stop_btn = st.button("Stop Webcam", on_click=toggle_webcam)

# Placeholder for video feed
frame_placeholder = st.empty()

# Webcam capture logic
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            st.session_state.run = False
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)
        time.sleep(0.1)  # Adjust frame rate
    cap.release()
else:
    frame_placeholder.empty()
