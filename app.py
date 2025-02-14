import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
import pyttsx3
from string import ascii_uppercase
import enchant

# إعداد المكونات الأساسية
model = load_model('cnn8grps_rad1_model.h5')
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
ddd = enchant.Dict("en-US")
offset = 29
engine = pyttsx3.init()
engine.setProperty("rate", 100)

# حالة التطبيق
if 'text' not in st.session_state:
    st.session_state.text = ""
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = ["", "", "", ""]

# فئة معالجة الفيديو
class SignLanguageTranslator(VideoTransformerBase):
    def __init__(self):
        self.white = np.ones((400, 400, 3), dtype=np.uint8) * 255

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # اكتشاف اليدين
        hands = hd.findHands(img, draw=False, flipType=True)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            img_crop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            
            if img_crop.size != 0:
                # معالجة الهيكل العظمي
                handz = hd2.findHands(img_crop, draw=False, flipType=True)
                if handz:
                    hand = handz[0]
                    pts = hand['lmList']
                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15
                    
                    # رسم الخطوط (نفس الكود الأصلي)
                    connections = [
                        (0, 4), (5, 8), (9, 12), (13, 16), (17, 20),
                        (5, 9), (9, 13), (13, 17), (0, 5), (0, 17)
                    ]
                    
                    for start, end in connections:
                        cv2.line(self.white, 
                                (pts[start][0]+os, pts[start][1]+os1),
                                (pts[end][0]+os, pts[end][1]+os1),
                                (0, 255, 0), 3)
                    
                    # التنبؤ بالحرف
                    white_input = cv2.resize(self.white, (400, 400))
                    prediction = model.predict(np.array([white_input]))[0]
                    ch1 = np.argmax(prediction)
                    
                    # إضافة الحرف للنص
                    st.session_state.text += chr(ch1 + 65)
                    
        return img

# واجهة المستخدم
st.title("Sign Language to Text - Live")

# مكون الكاميرا
ctx = webrtc_streamer(
    key="sign-language",
    video_transformer_factory=SignLanguageTranslator,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# عناصر التحكم
col1, col2 = st.columns(2)
with col1:
    if st.button("Speak Text"):
        engine.say(st.session_state.text)
        engine.runAndWait()
with col2:
    if st.button("Clear Text"):
        st.session_state.text = ""

# عرض النص والمقترحات
st.header("Translated Text")
st.write(st.session_state.text)

st.header("Suggestions")
if st.session_state.text:
    suggestions = ddd.suggest(st.session_state.text.split()[-1])[:4]
    cols = st.columns(4)
    for i, col in enumerate(cols):
        with col:
            if st.button(suggestions[i] if i < len(suggestions) else ""):
                st.session_state.text += " " + suggestions[i]
