import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Mohamed Reda | Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# ------------------------------
# Load Model
# ------------------------------
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))


    model_path = r"C:\Users\PC\Desktop\Computer Vision\Facial Emotion Recognition\CNN_FER13_Model.keras"
    # أو
    # model_path = os.path.join("C:", "Users", "PC", "Desktop", "Computer Vision", "Facial Emotion Recognition", "CNN_FER13_Model.keras")

    model = tf.keras.models.load_model(model_path)

    return model

model = load_model()

emotion_labels = {
    0: "Angry 😠",
    1: "Fearful 😨",
    2: "Happy 😄",
    3: "Surprised 😲",
    4: "Neutral 😐",
    5: "Sad 😢"
}


cascade_path = os.path.join(os.path.dirname(__file__),
                            "haarcascade_frontalface_default.xml")

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("❌ Cascade failed to load")
else:
    print("✅ Cascade loaded successfully")
    
# ------------------------------
# Helper Function
# ------------------------------
def predict_emotion(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.GaussianBlur(face_img, (1, 1), 0)
    face_img = cv2.convertScaleAbs(face_img, beta=120)
    face_img = face_img / 255.0
    face_img = np.expand_dims(np.expand_dims(face_img, -1), 0)
    
    prediction = model.predict(face_img)
    return emotion_labels[int(np.argmax(prediction))]

# ------------------------------
# UI
# ------------------------------
st.title("🎭 Real-Time Emotion Recognition System")
st.markdown("### CNN-Based Facial Emotion Classification")
st.markdown("---")

menu = st.sidebar.radio(
    "Choose Input Method",
    ["📷 Upload Image", "📸 Webcam"]
)

# ------------------------------
# Upload Image
# ------------------------------
if menu == "📷 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=3
        )

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            emotion = predict_emotion(roi)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                img,
                emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2
            )

        final_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(final_img, use_column_width=True)

# ------------------------------
# Webcam Mode
# ------------------------------
elif menu == "📸 Webcam":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=3
        )

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            emotion = predict_emotion(roi)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        FRAME_WINDOW.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

    camera.release()

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(
    "<center>Developed by <b>Mohamed Reda</b> | 2025</center>",
    unsafe_allow_html=True
)
