import numpy as np
import glob
import cv2
import random
faceDet1=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet2=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet3=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet4=cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
emotions=["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
fishface = cv2.face.FisherFaceRecognizer_create()
data={}
font = cv2.FONT_HERSHEY_SIMPLEX


def get_files(emotion):
	files=sorted(glob.glob("dataset/%s/*" %emotion))
	random.shuffle(files)
	training=files
	return training

def make_sets():
	training_data=[]
	training_labels=[]
	
	
	for emotion in emotions:
		training=get_files(emotion)
		count=0;
		for item in training:																											
			image=cv2.imread(item)
			gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			training_data.append(gray)
			training_labels.append(emotions.index(emotion))

	#print np.asarray(training_labels)
	#print np.asarray(prediction_labels)
	return training_data, training_labels

def run_recognizer():
	#training_data, training_labels=make_sets()
	print ("Now training the model...")
	#print np.asarray(training_labels)
	#print ("size of training set is:", len(training_labels), "images")
	#fishface.train(training_data, np.asarray(training_labels))
	fishface.read	("model.xml")
	#fishface.save("model.xml")
	print ("The model is successfully trained and saved!!")
	print ("Now predicting emotions...")
	count=0
	correct=0
	incorrect=0
	
    		
	for i in range(0, 5):
		cv2.namedWindow("testit")
		cam=cv2.VideoCapture(0)
		while(True):
			ret, framey=cam.read()
			cv2.imshow("testit", framey)
			if not ret:
				break
			k=cv2.waitKey(1)	
			if k%256 == 27:
				# ESC pressed
				print("Escape hit, closing...")
				break
			elif k%256 == 32:
				# SPACE pressed
				img_name = "test.png"
				cv2.imwrite(img_name, framey)
				print("Image Recorded")
				break

		cam.release()
		cv2.destroyAllWindows()
		frame=cv2.imread('test.png')
		gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#gray=cv2.imread("test.png", 0)
		#print "Inside nontry!!"
		face1=faceDet1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
		face2=faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
		face3=faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
		face4=faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
		print ("Printing data of various image detecting CasCades...")
		print  (face1)
		print  (face2)
		print  (face3)
		print  (face4)
		if len(face1)==1:
			facefeatures=face1
		elif len(face2)==1:
			facefeatures=face2
		elif len(face3)==1:
			facefeatures=face3
		elif len(face4)==1:
			facefeatures=face4
		else:
			facefeatures=""
			print ("NULL face")

		for (x, y, w, h) in facefeatures:
			print ("face found in file: test.png")
			gray=gray[y:y+h, x:x+w]
			try:
				#print "Inside try!!"
				out=cv2.resize(gray, (350, 350))
				cv2.imwrite("test.jpg", out)
			except:
				print ("There seems to be exception")
				pass

		image=cv2.imread("test.jpg")
		gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#gray=cv2.imread("test.jpg", 0)
		pred, confi=fishface.predict(gray)
		predicted="P:"+emotions[pred]
		#cv2.putText(gray, 'Incorrect',(5,25), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(gray, predicted,(5,310), font, 1, (255, 0, 0), 2, cv2.LINE_AA,False)
		cv2.putText(gray, 'Conf: '+str(float(("%0.2f"%(confi/1000)))),(5,340), font, 1, (255, 0, 0), 2, cv2.LINE_AA,False)
		cv2.imshow('image1', gray);
		cv2.waitKey(5000)
		cv2.destroyAllWindows()
	return 100

metascore=[]
for i in range(0, 1):
    correct=run_recognizer()
    #print "Had", correct, "precent correct!"
    metascore.append(correct)

#print "\n\nEnd result: ", np.mean(metascore), "percent correct!"
