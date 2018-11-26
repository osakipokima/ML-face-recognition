# -*- coding: utf-8 -*-
"""
Team: Matrix:Reloaded
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
	time_queue.pop()
	time_queue.insert(0,time.time())
	return '{:.10f}'.format(time_queue[0]-time_queue[1])


time_queue = [0,0]
time_dif()
time_display = True

#path = "profile.jpg"

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
#img = ImageTk.PhotoImage(Image.open(path))

#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
#panel = tk.Label(window, image = img)


# Font is a tuple of (font_family, size_in_points, style_modifier_string)


def clear_enterID():
	txt_enterID.delete(0, 'end')
	res = ""
	updateStatus(res)

def clear_enterName():
	txt_enterName.delete(0, 'end')
	res = ""
	updateStatus(res)

def is_number(s):
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
				cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
				if(time_display):print("11: \t" + str(time_dif()))
				#display the frame
				cv2.imshow('frame',img)
				if(time_display):print("12: \t" + str(time_dif()))
			#wait for 100 miliseconds
			if cv2.waitKey(20) & 0xFF == ord('q'):
				break
			# break if the sample number is morethan 100
			elif sampleNum>60:
				break
		if(time_display):print("13: \t" + str(time_dif()))
		cam.release()
		cv2.destroyAllWindows()
		if(time_display):print("14: \t" + str(time_dif()))
		res = "Images Saved for ID : " + Id +" Name : "+ name
		row = [Id , name]
		if(time_display):print("15: \t" + str(time_dif()))
		with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
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
	recognizer = cv2.face.LBPHFaceRecognizer_create()#recognizer = cv2.face_LBPHFaceRecognizer.create()#$cv2.createLBPHFaceRecognizer()
	if(time_display):print("19: \t" + str(time_dif()))
	harcascadePath = "haarcascade_frontalface_default.xml"
	detector = cv2.CascadeClassifier(harcascadePath)
	if(time_display):print("20: \t" + str(time_dif()))
	faces,Id = getImagesAndLabels("TrainingImage")
	if(time_display):print("20.5: \t" + str(time_dif()))
	recognizer.train(faces, np.array(Id))
	if(time_display):print("21: \t" + str(time_dif()))
	recognizer.save("TrainingImageLabel\Trainner.yml")
	if(time_display):print("21.5: \t" + str(time_dif()))
	res = "Image Trained"#+",".join(str(f) for f in Id)
	updateStatus(res)
	if(time_display):print("22: \t" + str(time_dif()))

def getImagesAndLabels(path):
	#get the path of all the files in the folder
	if(time_display):print("23: \t" + str(time_dif()))
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)]

	#create empth face list
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
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("TrainingImageLabel\Trainner.yml")
	if(time_display):print("26: \t" + str(time_dif()))
	harcascadePath = "haarcascade_frontalface_default.xml"
	if(time_display):print("27: \t" + str(time_dif()))
	faceCascade = cv2.CascadeClassifier(harcascadePath)
	if(time_display):print("28: \t" + str(time_dif()))
	df=pd.read_csv("StudentDetails\StudentDetails.csv")
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
				noOfFile=len(os.listdir("ImagesUnknown"))+1
				cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])
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
	fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
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
	

if(time_display):print("1: \t" + str(time_dif()))

# GUI stuff
window = tk.Tk()

Grid.rowconfigure(window, 0, weight=1)
Grid.columnconfigure(window, 0, weight=1)

window.title("Face Recogniser")
window.configure(background='maroon')

# creating a menu instance
menu = Menu(window)
window.config(menu=menu)
file = Menu(menu)
file.add_command(label="Exit", command=window.destroy )
menu.add_cascade(label="File", menu=file)

# ROW 0
title = tk.Label(window, text="Facial Recognition-Based Attendance Management System", bg="slate blue", font=('times', 30, 'italic bold underline'))
title.grid(row=0)


# ROW 1
frame_track = LabelFrame(window, text="Track")

trackImg = tk.Button(frame_track, text="Track Images", command=TrackImages, bg="slate blue", activebackground = "white", font=('times', 15, ' bold '))
trackImg.pack(side="left")

takeAttendance = tk.Button(frame_track, text="Take Attendance ", command=takeAttendance, bg="slate blue", activebackground = "white", font=('times', 15, ' bold '))
takeAttendance.pack(side="left")

frame_track.grid(row=1)


# ROW 2
frame_addUser = LabelFrame(window, text="Add User")

lbl_enterID = tk.Label(frame_addUser, text="Enter ID", bg="slate blue", font=('times', 15, ' bold ') )
lbl_enterID.grid(row=0, column=0)
txt_enterID = tk.Entry(frame_addUser, bg="slate blue", font=('times', 15, ' bold '))
txt_enterID.grid(row=0, column=1)
clearButton_enterID = tk.Button(frame_addUser, text="Clear", command=clear_enterID, bg="slate blue", activebackground = "white" ,font=('times', 15, ' bold '))
clearButton_enterID.grid(row=0, column=2)
takeImg = tk.Button(frame_addUser, text="Take Images", command=TakeImages, bg="slate blue", activebackground = "white" ,font=('times', 15, ' bold '))
takeImg.grid(row=0, column=3)

lbl_enterName = tk.Label(frame_addUser, text="Enter Name", bg="slate blue", font=('times', 15, ' bold '))
lbl_enterName.grid(row=1, column=0)
txt_enterName = tk.Entry(frame_addUser, bg="slate blue", font=('times', 15, ' bold ')  )
txt_enterName.grid(row=1, column=1)
clearButton_enterName = tk.Button(frame_addUser, text="Clear", command=clear_enterName, bg="slate blue", activebackground = "white" ,font=('times', 15, ' bold '))
clearButton_enterName.grid(row=1, column=2)
trainImg = tk.Button(frame_addUser, text="Train Images", command=TrainImages, bg="slate blue", activebackground = "white" ,font=('times', 15, ' bold '))
trainImg.grid(row=1, column=3)

frame_addUser.grid(row=2)

# ROW 3
frame_status = Frame(window)

lbl_status = tk.Label(frame_status, text="Status: ", bg="slate blue", font=('times', 15, ' bold '))
lbl_status.pack(side="left", anchor="w")

message_status = tk.Label(frame_status, text="", bg="slate blue", activebackground="yellow" ,font=('times', 15, ' bold '))
message_status.pack(side="left", anchor="w", fill="x")

frame_status.grid(row=3, sticky="w")

window.update()
window.minsize(window.winfo_width(), window.winfo_height())

#lbl_attendance = tk.Label(window, text="Attendance : ", bg="slate blue", font=('times', 15, ' bold '))
#lbl_attendance.place(x=400, y=650)

#message_attendance = tk.Label(window, text="", bg="slate blue", activeforeground = "green", font=('times', 15, ' bold '))
#message_attendance.place(x=700, y=650)





#quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="white"  ,bg="slate blue"  ,width=20  ,height=3, activebackground = "white" ,font=('times', 15, ' bold '))
#quitWindow.place(x=1100, y=500)

#copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
#copyWrite.tag_configure("superscript", offset=10)
#copyWrite.configure(state="disabled",fg="white"	 )
#copyWrite.pack(side="left")
#copyWrite.place(x=800, y=750)

if(time_display):print("2: \t" + str(time_dif()))
window.mainloop()
