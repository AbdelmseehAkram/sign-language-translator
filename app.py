import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
import pyttsx3
from string import ascii_uppercase

# Load model and setup utilities
model = load_model('cnn8grps_rad1_model.h5')
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
offset = 29
engine = pyttsx3.init()
engine.setProperty("rate", 100)

# Session state
if 'text' not in st.session_state:
    st.session_state.text = ""

# Set page config
st.set_page_config(page_title="Sign Language Translator", page_icon="🤟", layout="wide")

# Styling
st.markdown(
    """
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main-title {
            text-align: center;
            font-size: 40px;
            color: #4A90E2;
            font-weight: bold;
        }
        .sub-title {
            text-align: center;
            font-size: 20px;
            color: #666;
        }
        .stButton > button {
            width: 100%;
            padding: 10px;
            font-size: 18px;
            border-radius: 10px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #4A90E2;
            color: white;
        }
        .stText {
            font-size: 22px;
            font-weight: bold;
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<h1 class="main-title">🤟 Sign Language to Text</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Live translation of sign language into text</p>', unsafe_allow_html=True)

# Video Processing Class
class SignLanguageTranslator(VideoTransformerBase):
    def __init__(self):
        self.white = np.ones((400, 400, 3), dtype=np.uint8) * 255

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Hand detection
        hands = hd.findHands(img, draw=False, flipType=True)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            img_crop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            
            if img_crop.size != 0:
                # Skeleton processing
                handz = hd2.findHands(img_crop, draw=False, flipType=True)
                if handz:
                    hand = handz[0]
                    pts = hand['lmList']
                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15
                    
                    # Draw skeleton connections
                    connections = [
                        (0, 4), (5, 8), (9, 12), (13, 16), (17, 20),
                        (5, 9), (9, 13), (13, 17), (0, 5), (0, 17)
                    ]
                    
                    for start, end in connections:
                        cv2.line(self.white, 
                                (pts[start][0]+os, pts[start][1]+os1),
                                (pts[end][0]+os, pts[end][1]+os1),
                                (0, 255, 0), 3)
                    
                    # Predict letter
                    white_input = cv2.resize(self.white, (400, 400))
                    prediction = model.predict(np.array([white_input]))[0]
                    ch1 = np.argmax(prediction)
                    
                    # Append character to text
                    st.session_state.text += chr(ch1 + 65)
                    
        return img

# Camera Component
st.write("## 🎥 Live Camera Feed")
ctx = webrtc_streamer(
    key="sign-language",
    video_transformer_factory=SignLanguageTranslator,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Display Translated Text
st.write("## 📜 Translated Text")
st.markdown(f'<div class="stText">{st.session_state.text}</div>', unsafe_allow_html=True)

# Buttons for Actions
col1, col2 = st.columns(2)
with col1:
    if st.button("🗣️ Speak Text", key="speak"):
        engine.say(st.session_state.text)
        engine.runAndWait()
with col2:
    if st.button("🗑️ Clear Text", key="clear"):
        st.session_state.text = ""

# Additional Information
st.markdown("---")
st.write("### 💡 How to Use")
st.info("📌 Place your hand in front of the camera and make a sign for letters. The system will translate it in real time.")
st.success("✅ Click the **Speak Text** button to hear the translation.")
st.warning("⚠️ Ensure good lighting for better accuracy.")

# Footer
st.markdown("""
    <hr>
    <p style="text-align: center; color: #666;">💡 Sign Language Translator | Developed by <strong>Seha</strong> 🚀</p>
    """, unsafe_allow_html=True)
