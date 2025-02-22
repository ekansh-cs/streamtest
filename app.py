import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandTrackingTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)
    
    def transform(self, frame):
        # Convert frame to RGB
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        distance = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Get thumb and index finger tip coordinates
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate Euclidean distance
                distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

                # Display distance
                cv2.putText(img, f"Distance: {distance:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img

st.title("Hand Tracking with Pinch Detection")

# Stream webcam using WebRTC
webrtc_streamer(key="hand-tracking", video_transformer_factory=HandTrackingTransformer)

st.markdown("---")
st.info("Instructions:\n1. Allow camera access\n2. Show your hand to the camera\n3. The pinch distance will be displayed")
