from flask import Flask, render_template, Response, redirect
from scipy.spatial import distance
import numpy as np
from imutils import face_utils
import dlib
from camera import Video
import imutils
import cv2
import pygame

app=Flask(__name__)

# Tạo đối tượng khởi quay


@app.route('/')
def index():
	return render_template('index.html')


# Ước lượng tỷ lệ khung hình mắt
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

#Giới hạn	
frame_check = 10

# Sử dụng thư viên dlib để phát hiện khuôn mặt
detect = dlib.get_frontal_face_detector()

# Sử dụng thư viện dlib để dự đoán hình dạng thông qua tập dữ liệu
predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

# Điểm đầu và điểm cuối của mắt
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

#Khởi tạo và tải nhạc cảnh báo
pygame.mixer.init()
pygame.mixer.music.load("DEMO.mp3")


def drowsy_warning():
	flag=0
	cap = cv2.VideoCapture(0)

	while 1:
		if flag == frame_check:
			pygame.mixer.music.play()
		elif flag == 1:
			pygame.mixer.music.stop()

		#Bắt đầu đọc video
		ret,frame=cap.read()

		#Chọn kích thước cho khung hình
		frame = imutils.resize(frame, width=600)
		#chuyển đổi sang thang độ xám
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.putText(frame, "DROWSY DETECTION AND WARNING", (23,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,0), 2, cv2.LINE_AA, False)
		subjects = detect(gray, 0)
		for subject in subjects:
			shape = predictor(gray, subject)
			shape = face_utils.shape_to_np(shape)
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			# leftEyeHull = cv2.convexHull(leftEye)
			# rightEyeHull = cv2.convexHull(rightEye)
			# cv2.drawContours(frame, [leftEyeHull],  -1, (0, 0, 255), 1)
			# cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
			if ear < 0.2:
				flag += 1
				print (flag)
			else:
				flag = 0
		cv2.imshow("Frame", frame)
	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			# cap.release()
			# cv2.destroyAllWindows()
			break
	cap.release()
	cv2.destroyAllWindows()

# def detect_face(camera):
# 	while True:
# 		frame=camera.get_frame()

# 		yield(b'--frame\r\n'
# 					b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# # API nối kết đến hàm gen(Video)
# @app.route('/video')
# def video():
# 	return Response(detect_face(Video()), mimetype = 'multipart/x-mixed-replace; boundary=frame')


# API nối kết đến hàm drowsy_warning()
@app.route('/running')
def running():
	if drowsy_warning() is open:
		return Response(drowsy_warning(),mimetype = 'multipart/x-mixed-replace; boundary=frame')
	else:
		return redirect('/')

if __name__ == "__main__":
	app.run(debug=True)