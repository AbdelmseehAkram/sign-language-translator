import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model(r"model_alphabet_transfer.keras")
class_labels = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Streamlit UI setup
st.title("Sign Language Detection App")
st.markdown("**Press the 'Start Camera' button to begin detecting signs**")

# Camera toggle
run_camera = st.checkbox("Start Camera")

# Placeholder for displaying video frames
stframe = st.empty()

def preprocess_frame(frame):
    """Process the frame to match the model's input format."""
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame

if run_camera:
    # Start the webcam
    cap = cv2.VideoCapture(0)
    
    while run_camera:
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
        stframe.image(frame_rgb, channels="RGB")
    
    cap.release()
else:
    st.warning("Press the 'Start Camera' button to begin streaming.")
