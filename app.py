import av
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import gdown
import os

# Load the pre-trained model from Google Drive
model_url = "https://drive.google.com/file/d/1mnasQGJhxbxGW1wotIT1nr1icNzz9xdC/view?usp=sharing"  # استبدل YOUR_FILE_ID_HERE برابط Google Drive المناسب
model_path = "model_alphabet_transfer.keras"

# تنزيل الموديل إذا لم يكن موجودًا
if not os.path.exists(model_path):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(model_url, model_path, quiet=False)

model = load_model(model_path)
class_labels = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Streamlit UI setup
st.title("Sign Language Detection App")
st.markdown("**Press the button below to detect signs in real-time**")

# Video Transformer for sign detection
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.class_labels = class_labels

    def preprocess_frame(self, frame):
        """Process the frame to match the model's input format."""
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)
        return input_frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert the frame colors from BGR to RGB
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the frame and make predictions
        input_frame = self.preprocess_frame(frame_rgb)
        predictions = self.model.predict(input_frame)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Display the predictions on the frame
        label = f"{self.class_labels[predicted_class]} ({confidence:.2f})"
        cv2.putText(frame_rgb, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

# Start the WebRTC streamer with media constraints to ensure compatibility
webrtc_streamer(
    key="sign-language-detector", 
    video_transformer_factory=SignLanguageTransformer, 
    media_stream_constraints={"video": True, "audio": False}
)
