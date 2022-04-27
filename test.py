from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
#Giới hạn	
frame_check = 10

#Sử dụng thư viện dlib để nhận dạng mặt
detect = dlib.get_frontal_face_detector()
#Sử dụng thư viện dlib để  dự đoán hình dạng thông qua tập dữ liệu "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

pygame.mixer.init()
pygame.mixer.music.load("DEMO.mp3")
#Tạo đối tượng quay video
cap = cv2.VideoCapture(0)

#Khai báo cờ đếm số lần nhắm mắt
flag=0
    
while True:
	if flag == frame_check:
		pygame.mixer.music.play()
	elif flag == 1:
		pygame.mixer.music.stop()
	#Bắt đầu đọc video
	ret, frame=cap.read()

	#Chọn kích thước cho khung hình
	frame = imutils.resize(frame, width=600)

	#chuyển đổi sang thang độ xám
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.putText(frame, "DROWSY DETECTION AND WARNING", (23,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA, False)
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
		# cv2.drawContours(frame, [leftEyeHull],  -1, (0, 0, 255), 3)
		# cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 3)
		if ear < 0.2:
			flag += 1
			print (flag)
		else:
			flag = 0
	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
cv2.destroyAllWindows()
cap.release() 