# -*- coding: utf-8 -*-
"""
Team: Matrix:Reloaded

Attendance: Applied Facial Recognition
"""

import tkinter as tk
from tkinter import Message ,Text, Menu, LabelFrame, Frame, Grid
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

def time_dif():
	""" Computes time difference. Primarly for optimization """
	time_queue.pop()
	time_queue.insert(0,time.time())
	return '{:.10f}'.format(time_queue[0]-time_queue[1])


time_queue = [0,0]
time_dif()
time_display = True


def clear_enterID():
	""" Reset the GUI ID entry box """
	txt_enterID.delete(0, 'end')
	res = ""
	updateStatus(res)

def clear_enterName():
	""" Reset the GUI name entry box """
	txt_enterName.delete(0, 'end')
	res = ""
	updateStatus(res)

def is_number(s):
	""" Number validation check """
	try:
		float(s)
		return True
	except ValueError:
		pass

	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass

	return False

def TakeImages():
	""" Input validation of ID and Name in the GUI and add to csv
		Open camera begin gathering 60 samples
		When a face is in the view of the camera it will assign it to the ID and name in the GUI
		Save data in gray scale label images as ID number with sample number """
	Id=(txt_enterID.get())
	name=(txt_enterName.get())

	if (Id == ""):
		updateStatus("ID must NOT be blank")
		return
	if (not is_number(Id)):
		updateStatus("ID MUST be a number")
		return
	if (name == ""):
		updateStatus("Name must NOT be blank")
		return

	updateStatus("Taking Images") # this status update doesnt work for some resson

	if(time_display):print("3: \t" + str(time_dif()))

	if(time_display):print("4: \t" + str(time_dif()))
	if(is_number(Id) and name.isalpha()):
		cam = cv2.VideoCapture(0)
		if(time_display):print("5: \t" + str(time_dif()))
		harcascadePath = "haarcascade_frontalface_default.xml"
		detector=cv2.CascadeClassifier(harcascadePath)
		if(time_display):print("6: \t" + str(time_dif()))
		sampleNum=0
		while(True):
			ret, img = cam.read()
			if(time_display):print("7: \t" + str(time_dif()))
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			if(time_display):print("8: \t" + str(time_dif()))
			faces = detector.detectMultiScale(gray, 1.3, 5)
			if(time_display):print("9: \t" + str(time_dif()))
			for (x,y,w,h) in faces:
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
				if(time_display):print("10: \t" + str(time_dif()))
				#incrementing sample number
				sampleNum=sampleNum+1
				#saving the captured face in the dataset folder TrainingImage
				cv2.imwrite("TrainingImage/ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
				if(time_display):print("11: \t" + str(time_dif()))
				#display the frame
				cv2.imshow('frame',img)
				if(time_display):print("12: \t" + str(time_dif()))
			#wait for 100 miliseconds
			if cv2.waitKey(20) & 0xFF == ord('q'):
				break
			# break if the sample number is morethan 60
			elif sampleNum>60:
				break
		if(time_display):print("13: \t" + str(time_dif()))
		cam.release()
		cv2.destroyAllWindows()
		if(time_display):print("14: \t" + str(time_dif()))
		res = "Images Saved for ID : " + Id +" Name : "+ name
		row = [Id , name]
		if(time_display):print("15: \t" + str(time_dif()))
		with open('StudentDetails/StudentDetails.csv','a+') as csvFile:
			writer = csv.writer(csvFile)
			if(time_display):print("16: \t" + str(time_dif()))
			writer.writerow(row)
			if(time_display):print("17: \t" + str(time_dif()))
		csvFile.close()
		if(time_display):print("18: \t" + str(time_dif()))
		updateStatus(res)
	else:
		if(is_number(Id)):
			res = "Enter Alphabetical Name"
			updateStatus(res)
		if(name.isalpha()):
			res = "Enter Numeric Id"
			updateStatus(res)

def TrainImages():
	""" Local Binary Pattern Haar Face Recognizer (LBPHFaceRecognizer)
		Data given to the trainner are expected to be grayscale
		Faces are saved with the ID inputted in the GUI
		The analyzed data is serialized and saved into Trainner.yml for future use """
	recognizer = cv2.face.LBPHFaceRecognizer_create()#recognizer = cv2.face_LBPHFaceRecognizer.create()#$cv2.createLBPHFaceRecognizer()
	if(time_display):print("19: \t" + str(time_dif()))
	harcascadePath = "haarcascade_frontalface_default.xml"
	detector = cv2.CascadeClassifier(harcascadePath)
	if(time_display):print("20: \t" + str(time_dif()))
	faces,Id = getImagesAndLabels("TrainingImage")
	if(time_display):print("20.5: \t" + str(time_dif()))
	recognizer.train(faces, np.array(Id))
	if(time_display):print("21: \t" + str(time_dif()))
	if (not os.path.isdir("TrainingImageLabel")): os.mkdir("TrainingImageLabel")
	recognizer.save("TrainingImageLabel/Trainner.yml")
	if(time_display):print("21.5: \t" + str(time_dif()))
	res = "Image Trained"#+",".join(str(f) for f in Id)
	updateStatus(res)
	if(time_display):print("22: \t" + str(time_dif()))

def getImagesAndLabels(path):
	""" Gather the path names for all the photos saved in the trainingImage folder
		Convert all data to grayscale for use in the Haar algorithm
		Convert each image into a numpy array
		Crop image to only contain the detected face """
	#get the path of all the files in the folder
	if(time_display):print("23: \t" + str(time_dif()))
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)]

	#create empty face list
	faces=[]
	#create empty ID list
	Ids=[]
	#now looping through all the image paths and loading the Ids and the images
	if(time_display):print("24: \t" + str(time_dif()))
	for imagePath in imagePaths:
		#loading the image and converting it to gray scale
		pilImage=Image.open(imagePath).convert('L')
		#Now we are converting the PIL image into numpy array
		imageNp=np.array(pilImage,'uint8')
		#getting the Id from the image
		Id=int(os.path.split(imagePath)[-1].split(".")[1])
		# extract the face from the training image sample
		faces.append(imageNp)
		Ids.append(Id)
		if(time_display):print("25: \t" + str(time_dif()))
	return faces,Ids

def TrackImages():
	""" Read the yml file containing the analyzed data
		Open camera to begin tracking faces
		Apply box around detected faces and add ID and name if the face is recognized
		When a face is recognized add an entry for it into the attendance csv """
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("TrainingImageLabel/Trainner.yml")
	if(time_display):print("26: \t" + str(time_dif()))
	harcascadePath = "haarcascade_frontalface_default.xml"
	if(time_display):print("27: \t" + str(time_dif()))
	faceCascade = cv2.CascadeClassifier(harcascadePath)
	if(time_display):print("28: \t" + str(time_dif()))
	df=pd.read_csv("StudentDetails/StudentDetails.csv")
	if(time_display):print("29: \t" + str(time_dif()))
	cam = cv2.VideoCapture(0)
	if(time_display):print("30: \t" + str(time_dif()))
	font = cv2.FONT_HERSHEY_SIMPLEX
	col_names =  ['Id','Name','Date','Time']
	attendance = pd.DataFrame(columns = col_names)
	if(time_display):print("31: \t" + str(time_dif()))
	while True:
		ret, im =cam.read()
		if(time_display):print("32: \t" + str(time_dif()))
		gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		if(time_display):print("33: \t" + str(time_dif()))
		faces=faceCascade.detectMultiScale(gray, 1.2,5)
		if(time_display):print("34: \t" + str(time_dif()))
		for(x,y,w,h) in faces:
			cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
			if(time_display):print("35: \t" + str(time_dif()))
			Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
			if(time_display):print("36: \t" + str(time_dif()))
			if(conf < 50):
				ts = time.time()
				date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
				timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
				aa=df.loc[df['Id'] == Id]['Name'].values
				tt=str(Id)+"-"+aa
				attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
				if(time_display):print("37: \t" + str(time_dif()))

			else:
				Id='Unknown'
				tt=str(Id)
				if(time_display):print("38: \t" + str(time_dif()))
			if(conf > 75):
				if (not os.path.isdir("ImagesUnknown")): os.mkdir("ImagesUnknown")
				noOfFile=len(os.listdir("ImagesUnknown"))+1
				cv2.imwrite("ImagesUnknown/Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])
				if(time_display):print("39: \t" + str(time_dif()))
			cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)
		attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
		if(time_display):print("40: \t" + str(time_dif()))
		cv2.imshow('im',im)
		if(time_display):print("41: \t" + str(time_dif()))
		if (cv2.waitKey(1)==ord('q')):
			break
	if(time_display):print("42: \t" + str(time_dif()))
	ts = time.time()
	date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
	timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
	Hour,Minute,Second=timeStamp.split(":")
	fileName="Attendance/Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
	attendance.to_csv(fileName,index=False)
	cam.release()
	cv2.destroyAllWindows()
	#print(attendance)
	res=attendance
	message_attendance.configure(text= res)
	if(time_display):print("43: \t" + str(time_dif()))

def takeAttendance():
	print("TODO")

def updateStatus(status):
	message_status.configure(text=status)

def resizePadding(event):
	print ("Width: ", event.width, "Height: ", event.height)


if(time_display):print("1: \t" + str(time_dif()))

# GUI stuff
'''
main window{
	title{
		title frame
	}
	frame_mainContent{
		frame_track{
			trackFaces | takeAttendance
		}
		frame_addUser{
			frame_enterData{
				lbl_enterID   | txt_enterID   | clearButton_enterID
				lbl_enterName | txt_enterName | clearButton_enterName
			}
			frame_trainMachine{
			takeImg
			trainImg
			}
		}
	}
	frame_status{
		lbl_status | lbl_status
	}

}
'''

""" GUI implementation """
window = tk.Tk()

window.title("Face Recogniser")
#window.configure(background='snow')

# creating a menu instance
menu = Menu(window)
window.config(menu=menu)
file = Menu(menu)
file.add_command(label="Exit", command=window.destroy )
menu.add_cascade(label="File", menu=file)

# Font is a tuple of (font_family, size_in_points, style_modifier_string)

# window ROW 0
title = tk.Label(window, text="Facial Recognition-Based Attendance Management System", bg="#7272d8", font=('times', 30, 'italic bold underline'))
title.grid(row=0, sticky="NS", ipadx=10)



# window ROW 1
frame_mainContent = Frame(window)

# frame_mainContent ROW 0
frame_track = LabelFrame(frame_mainContent, text="Track")

trackFaces = tk.Button(frame_track, text="Track Faces", command=TrackImages, font=('times', 15, ' bold '))
trackFaces.grid(row=0, column=0)

takeAttendance = tk.Button(frame_track, text="Take Attendance", command=takeAttendance, font=('times', 15, ' bold '))
takeAttendance.grid(row=0, column=1)

frame_track.grid(row=0, ipady=10, ipadx=10)
# Center buttons in frame
frame_track.grid_rowconfigure(0, weight=1)
frame_track.grid_columnconfigure(0, weight=1)
frame_track.grid_columnconfigure(1, weight=1)

# frame_mainContent ROW 1
frame_addUser = LabelFrame(frame_mainContent, text="Add User")

frame_enterData = Frame(frame_addUser)
# frame_enterData ROW 0
lbl_enterID = tk.Label(frame_enterData, text="Enter ID", font=('times', 15, ' bold ') )
lbl_enterID.grid(row=0, column=0, sticky="W")
txt_enterID = tk.Entry(frame_enterData, font=('times', 15, ' bold '))
txt_enterID.grid(row=0, column=1)
clearButton_enterID = tk.Button(frame_enterData, text="Clear", command=clear_enterID,font=('times', 15, ' bold '))
clearButton_enterID.grid(row=0, column=2)
# frame_enterData ROW 1
lbl_enterName = tk.Label(frame_enterData, text="Enter Name", font=('times', 15, ' bold '))
lbl_enterName.grid(row=1, column=0, sticky="W")
txt_enterName = tk.Entry(frame_enterData, font=('times', 15, ' bold ')  )
txt_enterName.grid(row=1, column=1)
clearButton_enterName = tk.Button(frame_enterData, text="Clear", command=clear_enterName, font=('times', 15, ' bold '))
clearButton_enterName.grid(row=1, column=2)
frame_enterData.grid(row=0, column=0)
frame_enterData.grid_rowconfigure(0, weight=1)

frame_trainMachine = Frame(frame_addUser)
# frame_trainMachine ROW 0
takeImg = tk.Button(frame_trainMachine, text="Take Images", command=TakeImages, bg='light goldenrod', font=('times', 15, ' bold '), padx=5, pady=5)
takeImg.grid(row=0, pady=(3,0))
# frame_trainMachine ROW 1
trainImg = tk.Button(frame_trainMachine, text="Train Images", command=TrainImages, bg='tomato', font=('times', 15, ' bold '), padx=5, pady=5)
trainImg.grid(row=1, pady=(0,4))
frame_trainMachine.grid(row=0, column=1, padx=(5,0))
frame_trainMachine.grid_rowconfigure(0, weight=1)

frame_addUser.grid_rowconfigure(0, weight=1)
frame_addUser.grid_columnconfigure(0, weight=1)
frame_addUser.grid_columnconfigure(1, weight=1)

frame_addUser.grid(row=1, ipady=5, ipadx=50)


frame_mainContent.grid(row=1)
# Expand mainContent to whole window
frame_mainContent.grid(sticky="EW")
# Center mainContent within window
frame_mainContent.grid_rowconfigure(0, weight=1)
frame_mainContent.grid_rowconfigure(1, weight=1)
frame_mainContent.grid_columnconfigure(0, weight=1)


# window ROW 2
frame_status = Frame(window)

lbl_status = tk.Label(frame_status, text="Status: ", bg="#ebebeb", font=('times', 15, ' bold '))
lbl_status.pack(side="left", anchor="w")

message_status = tk.Label(frame_status, text="", bg="#ebebeb", font=('times', 15, ' bold '))
message_status.pack(side="left", anchor="w", fill="x")

frame_status.grid(row=2, sticky="sw")

# window resize config
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=3)
window.grid_columnconfigure(0, weight=1)

#window set min size
window.update()
default_width = window.winfo_width()
default_height = window.winfo_height()
window.minsize(default_width, default_height)

#window.bind('<Configure>', resizePadding)

if(time_display):print("2: \t" + str(time_dif()))
window.mainloop()
