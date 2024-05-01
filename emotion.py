from keras.models import load_model
import cv2
import numpy as np

emotion_dict = {0: "angry", 1: "disgusted", 2: "fear", 
				3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}


emotion_model = load_model(r'C:\Users\Administrator\Desktop\as\emotion_detection\emoo_model.h5')
emotion_model.summary()


cap = cv2.VideoCapture(0)
while True:
	# Find haar cascade to draw bounding box around face
	ret, frame = cap.read()
	frame = cv2.resize(frame, (1280, 720))
	if not ret:
		print(ret)
	# Create a face detector
	face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces available on camera
	num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

	# take each face available on the camera and Preprocess it
	for (x, y, w, h) in num_faces:
		cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
		roi_gray_frame = gray_frame[y:y + h, x:x + w]
		cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

		# predict the emotions
		emotion_prediction = emotion_model.predict(cropped_img)
		maxindex = int(np.argmax(emotion_prediction))
		cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

	cv2.imshow('Emotion Detection', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
