import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to numpy array and return without modification
        return frame.to_ndarray(format="bgr24")

st.title("Webcam Stream")

# Stream webcam using WebRTC
webrtc_streamer(key="video-stream", video_transformer_factory=VideoTransformer)

st.markdown("---")
st.info("Instructions:\n1. Allow camera access\n2. The webcam feed will be displayed")
