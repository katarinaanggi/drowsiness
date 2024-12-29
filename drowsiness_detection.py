import streamlit as st
import cv2
import numpy as np
import pygame
import os
from tensorflow.keras.models import load_model
import time  # Tambahkan modul time untuk durasi

# Load model CNN
MODEL_PATH = "dataset/DatasetFinal/model/fcnn_model.h5"  # Ganti dengan model Anda
model = load_model(MODEL_PATH)

# Threshold dan konfigurasi deteksi
EYE_CLOSED_THRESHOLD = 0.5  # Probabilitas threshold untuk mata tertutup
CLOSED_FRAMES_THRESHOLD = 30  # Jumlah frame mata tertutup untuk mendeteksi kantuk

# Initialize pygame mixer
pygame.mixer.init()

# Fungsi untuk memproses gambar mata
def preprocess_eye(eye_image):
    eye_image = cv2.resize(eye_image, (64, 64))  # Resize sesuai kebutuhan
    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)  # Ubah ke grayscale
    eye_image = eye_image.astype("float") / 255.0  # Normalisasi
    eye_image = eye_image.flatten()  # Ratakan array menjadi 1D
    eye_image = np.expand_dims(eye_image, axis=0)  # Tambahkan batch dimension
    return eye_image

# Function to process image
def process_image(image, face_cascade, eye_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    closed_frames = 0  # Track the number of frames with closed eyes

    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face)

        # Ensure eyes are detected before processing
        if len(eyes) == 0:
            continue  # Skip if no eyes are detected

        for (ex, ey, ew, eh) in eyes:
            eye = face[ey:ey + eh, ex:ex + ew]
            processed_eye = preprocess_eye(eye)
            prediction = model.predict(processed_eye)
            prob_closed = prediction[0][0]  # Probabilitas mata tertutup

            if prob_closed > EYE_CLOSED_THRESHOLD:
                closed_frames += 1
                color = (0, 0, 255)  # Merah jika mata tertutup
                status = "Closed"
            else:
                color = (0, 255, 0)  # Hijau jika mata terbuka
                status = "Open"

            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), color, 2)
            cv2.putText(face, f"{status}: {prob_closed:.2f}", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, closed_frames

# Handle sound playing with pygame.mixer
def play_alarm_sound():
    audio_file = 'alarm.wav'  # Path to your sound file
    if os.path.exists(audio_file):
        try:
            pygame.mixer.music.load(audio_file)  # Load the sound file
            pygame.mixer.music.play()  # Play sound
        except Exception as e:
            st.error(f"Error playing sound: {e}")
    else:
        st.error("Sound file not found!")

def drowsiness_detection_page():
    # Streamlit Sidebar Menu
    menu_options = ['Webcam', 'Upload Image']
    selected_option = st.sidebar.selectbox("Choose Input Type", menu_options)

    # Streamlit UI
    st.markdown("""<div style="text-align: center; color: #008080; font-size: 40px; font-weight: bold;">
        <h1>Drowsiness Detection System</h1>
        <p>Real-time video feed to detect drowsiness using CNN.</p>
    </div>""", unsafe_allow_html=True)

    # Initialize the cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # Initialize variables
    closed_frames = 0
    alarm_playing = False  # Track alarm state
    alarm_start_time = None  # Track alarm start time

    # Webcam Input Handling
    if selected_option == 'Webcam':
        run = st.checkbox("Start Detection")
        FRAME_WINDOW = st.image([])  # Initialize the image window

        # Initialize video capture
        cap = None
        if run:
            cap = cv2.VideoCapture(0)  # Use webcam
            if not cap.isOpened():
                st.error("Failed to open webcam.")
                return

        while run:
            if cap is None:
                break

            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:  # Skip if no faces are detected
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                continue

            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(face)

                if len(eyes) == 0:  # Skip if no eyes are detected
                    continue

                for (ex, ey, ew, eh) in eyes:
                    eye = face[ey:ey + eh, ex:ex + ew]
                    processed_eye = preprocess_eye(eye)
                    prediction = model.predict(processed_eye)
                    prob_closed = prediction[0][0]  # Probability of closed eyes

                    # Check for drowsiness
                    if prob_closed > EYE_CLOSED_THRESHOLD:
                        closed_frames += 1
                        color = (0, 0, 255)  # Red for closed eyes
                        status = "Closed"
                    else:
                        closed_frames = 0
                        color = (0, 255, 0)  # Green for open eyes
                        status = "Open"

                    # Draw rectangle around eyes and display the status
                    cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), color, 2)
                    cv2.putText(face, f"{status}: {prob_closed:.2f}", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Alert for drowsiness when eyes are closed for too long
                    if closed_frames > CLOSED_FRAMES_THRESHOLD:
                        cv2.putText(frame, "Drowsiness Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        if not alarm_playing:
                            play_alarm_sound()  # Play the alarm sound
                            alarm_playing = True
                            alarm_start_time = time.time()  # Record start time

            # Stop alarm after 4 seconds
            if alarm_playing and time.time() - alarm_start_time > 4:
                pygame.mixer.music.stop()
                alarm_playing = False
                alarm_start_time = None  # Reset alarm start time

            # Display video feed with face and eye detection
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Release video capture when stopped
        if cap is not None:
            cap.release()
            FRAME_WINDOW.image([])  # Clear video feed

    # Handle image upload
    elif selected_option == 'Upload Image':
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            processed_image, _ = process_image(image, face_cascade, eye_cascade)

            st.image(processed_image, channels="BGR", caption="Processed Image", use_column_width=True)
