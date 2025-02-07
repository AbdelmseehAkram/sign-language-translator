import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model(r"model_alphabet_transfer.keras")
class_labels = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Streamlit UI setup
st.title("Sign Language Detection App")
st.markdown("**Press the buttons below to control the camera and detect signs in real-time**")

# UI controls
start_camera = st.button("Start Camera")
stop_camera = st.button("Stop Camera")
status_placeholder = st.empty()
frame_placeholder = st.empty()

# Define camera state
camera_active = False

# Function to preprocess frame for the model
def preprocess_frame(frame):
    """Process the frame to match the model's input format."""
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame

# Camera processing logic
if start_camera:
    camera_active = True
    cap = cv2.VideoCapture(0)
    status_placeholder.success("Camera Started!")

if stop_camera:
    camera_active = False
    status_placeholder.warning("Camera Stopped!")

# Loop for real-time camera feed
while camera_active:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to read from the camera!")
        break

    # Convert the frame colors from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and make predictions
    input_frame = preprocess_frame(frame_rgb)
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display the predictions on the frame
    label = f"{class_labels[predicted_class]} ({confidence:.2f})"
    cv2.putText(frame_rgb, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame in Streamlit
    frame_placeholder.image(frame_rgb, channels="RGB")

if camera_active:
    cap.release()
