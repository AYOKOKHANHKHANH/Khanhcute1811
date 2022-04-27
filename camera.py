import cv2

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
class Video(object):
	def __init__(self):
		self.cap=cv2.VideoCapture(0)
	def __del__(self):
		self.cap.release()
	def get_frame(self):
		ret,frame=self.cap.read()
		faces=faceDetect.detectMultiScale(frame, 1.3, 5)
		for x,y,w,h in faces:
			x1,y1 = x+w, y+h
			cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 1)
			cv2.line(frame, (x,y), (x+30, y), (255,0,255), 6)
			cv2.line(frame, (x,y), (x, y+30), (255,0,255), 6) # Trên, trái

			cv2.line(frame, (x1,y), (x1-30, y), (255,0,255), 6) 
			cv2.line(frame, (x1,y), (x1, y+30), (255,0,255), 6) # Trên, phải

			cv2.line(frame, (x,y1), (x+30, y1), (255,0,255), 6)
			cv2.line(frame, (x,y1), (x,y1-30), (255,0,255), 6) # Dưới, phải

			cv2.line(frame, (x1,y1), (x1-30, y1), (255,0,255), 6)
			cv2.line(frame, (x,y1), (x, y1-30), (255,0,255), 6) # Dưới, phải


			

		ret,jpg=cv2.imencode('.jpg',frame)
		return jpg.tobytes()