import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Webcam with Drawing Feature")

enable = st.checkbox("Enable camera")

# Define a class to handle video processing
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.drawing = False
        self.last_x, self.last_y = None, None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # If drawing is enabled, add functionality
        def draw(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.last_x, self.last_y = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    cv2.line(img, (self.last_x, self.last_y), (x, y), (0, 255, 0), 3)
                    self.last_x, self.last_y = x, y

            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False

        cv2.setMouseCallback("Webcam Feed", draw)

        return frame.from_ndarray(img, format="bgr24")

if enable:
    webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer)
