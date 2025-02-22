import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)

# Initialize session state
if 'webcam_on' not in st.session_state:
    st.session_state.webcam_on = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

def detect_pinch(frame):
    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    distance = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Get thumb and index finger tip coordinates
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate Euclidean distance
            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

            # Display distance on the frame
            cv2.putText(frame, f"Distance: {distance:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame, distance

st.title("Webcam Stream")

# Control buttons
col1, col2 = st.columns(2)
with col1:
    start_btn = st.button("Start Webcam")
with col2:
    stop_btn = st.button("Stop Webcam")

# Toggle webcam
if start_btn and not st.session_state.webcam_on:
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.webcam_on = True
if stop_btn and st.session_state.webcam_on:
    st.session_state.cap.release()
    st.session_state.webcam_on = False

# Display webcam feed
if st.session_state.webcam_on:
    image_placeholder = st.empty()
    distance_placeholder = st.empty()

    while st.session_state.webcam_on:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break

        # Process frame
        processed_frame, distance = detect_pinch(frame)
        
        # Display distance in the sidebar
        if distance is not None:
            distance_placeholder.markdown(f"**Current Pinch Distance:** {distance:.4f}")
        else:
            distance_placeholder.markdown("**No hands detected**")
        
        # Display frame
        image_placeholder.image(processed_frame, channels="BGR", width=640)

# Cleanup when stopping
if not st.session_state.webcam_on and st.session_state.cap is not None:
    st.session_state.cap.release()

st.markdown("---")
st.info("Instructions:\n1. Click 'Start Webcam' to begin\n2. Show your hand to the camera\n3. The distance between your thumb and index finger will be displayed")
