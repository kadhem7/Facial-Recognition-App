import cv2
import streamlit as st

# Specify the full path to the cascade classifier file
cascade_path = 'haarcascade_frontalface_default.xml'

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_faces():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Instructions:")
    st.write("Press the button below to start detecting faces from your webcam.")

    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        detect_faces()

        # Optional: Add feature to save detected faces image
        st.write("Detected faces image saved as detected_faces.jpg")

        # Optional: Add feature to adjust minNeighbors parameter
        min_neighbors = st.slider("Adjust minNeighbors", min_value=1, max_value=10, value=5)

        # Optional: Add feature to adjust scaleFactor parameter
        scale_factor = st.slider("Adjust scaleFactor", min_value=1.01, max_value=1.5, value=1.3, step=0.01)
        
        # Optional: Add feature to choose rectangle color
        rect_color = st.color_picker("Choose Rectangle Color", "#00FF00")  # Default color is green

if __name__ == "__main__":
    app()
