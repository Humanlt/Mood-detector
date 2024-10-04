import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# face detection model (Haar Cascade)
face_detector = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')


# emotion recognition model
emotion_model = load_model('emotion_model.h5')

#  emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# vedio capture 
cap = cv2.VideoCapture(0)

# webcam is opened correctly? 
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# frame processing 
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # read frame
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # gray image 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find face 
    faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(48, 48)# remember to modify the size of fave
    )

    # loop 
    for (x, y, w, h) in faces:

        roi_gray = gray_frame[y:y + h, x:x + w]
        # color
        roi_color = frame[y:y + h, x:x + w]

        roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # preprocess face image
        roi = roi_gray_resized.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # emotion prediction
        prediction = emotion_model.predict(roi)[0]
        emotion_probability = np.max(prediction)
        emotion_label = emotion_labels[prediction.argmax()]

        # D
        # emotion label and bounding box
        label_text = f"{emotion_label} ({emotion_probability*100:.2f}%)"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # display the resulting frame
    cv2.imshow('Mood Detection', frame)

    # q key out 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release capture 
cap.release()
cv2.destroyAllWindows()
