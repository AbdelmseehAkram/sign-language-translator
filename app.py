import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
from keras.models import load_model
import pyttsx3
import enchant
import os

# حل مشكلة libGL.so.1
os.system("apt-get update && apt-get install -y libgl1-mesa-glx")

# تحميل النموذج
model = load_model('cnn8grps_rad1_model.h5')

ddd = enchant.Dict("en-US")
offset = 29
engine = pyttsx3.init()
engine.setProperty("rate", 100)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def preprocess_image(image):
    image = image.convert("L").resize((400, 400))  # تحويل لصورة رمادية وتغيير الحجم
    return np.array(image).reshape(1, 400, 400, 1) / 255.0  # تحويل إلى مصفوفة وتطبيع القيم

# حالة التطبيق
if 'text' not in st.session_state:
    st.session_state.text = ""
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = ["", "", "", ""]

# فئة معالجة الفيديو
class SignLanguageTranslator(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="rgb24")
        results = hands.process(img)
        
        if results.multi_hand_landmarks:
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            
            for hand_landmarks in results.multi_hand_landmarks:
                points = [(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in hand_landmarks.landmark]
                
                connections = [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20), (5, 9), (9, 13), (13, 17), (0, 5), (0, 17)]
                for start, end in connections:
                    draw.line([points[start], points[end]], fill=(0, 255, 0), width=3)
            
            white_input = preprocess_image(img_pil)
            prediction = model.predict(white_input)[0]
            ch1 = np.argmax(prediction)
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
