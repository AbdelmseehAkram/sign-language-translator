import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
from string import ascii_uppercase

# Initialize core components
model = load_model('cnn8grps_rad1_model.h5')
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
offset = 29

# Application state
if 'text' not in st.session_state:
    st.session_state.text = ""
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = ["", "", "", ""]

# Video processing class
class SignLanguageTranslator(VideoTransformerBase):
    def __init__(self):
        self.white = np.ones((400, 400, 3), dtype=np.uint8) * 255

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Detect hands
        hands = hd.findHands(img, draw=False, flipType=True)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            img_crop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            
            if img_crop.size != 0:
                # Process skeleton
                handz = hd2.findHands(img_crop, draw=False, flipType=True)
                if handz:
                    hand = handz[0]
                    pts = hand['lmList']
                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15
                    
                    # Draw lines
                    connections = [
                        (0, 4), (5, 8), (9, 12), (13, 16), (17, 20),
                        (5, 9), (9, 13), (13, 17), (0, 5), (0, 17)
                    ]
                    
                    for start, end in connections:
                        cv2.line(self.white, 
                                (pts[start][0]+os, pts[start][1]+os1),
                                (pts[end][0]+os, pts[end][1]+os1),
                                (0, 255, 0), 3)
                    
                    # Predict character
                    white_input = cv2.resize(self.white, (400, 400))
                    prediction = model.predict(np.array([white_input]))[0]
                    ch1 = np.argmax(prediction)
                    
                    # Add character to text
                    st.session_state.text += chr(ch1 + 65)
                    
        return img

# UI Components
st.title("Sign Language to Text - Live")

# Camera component
ctx = webrtc_streamer(
    key="sign-language",
    video_transformer_factory=SignLanguageTranslator,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Control buttons
if st.button("Clear Text"):
    st.session_state.text = ""

# Display translated text
st.header("Translated Text")
st.write(st.session_state.text)
