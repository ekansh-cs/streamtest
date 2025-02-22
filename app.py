import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Webcam Stream")

# Stream webcam without processing
webrtc_streamer(key="video-stream")

st.markdown("---")
st.info("Instructions:\n1. Allow camera access\n2. The webcam feed will be displayed")
