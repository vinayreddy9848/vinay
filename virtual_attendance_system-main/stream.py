import numpy as np
import pandas as pd

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from webcam import webcam
import requests
import io
from datetime import datetime
import cv2
captureimg_list=[]

def Registrationcamera_main():
	st.title("Registration")

	Username = st.text_input("USERNAME")
	print("Username", Username)

	empId = st.text_input("Employee ID")
	empdf = pd.read_csv('app/main/static/data/empdata.csv', header=None)
	temp = 0
	if len(empId)!=0:

		try:
			empId = int(empId)
			temp = 1
			if empId not in list(empdf[0]):
				st.write('Please enter Valid employeeID')
				temp = 0
		except:
			st.write('Please write only integer value')



	#st.subheader("Press start to open the camera")
	#webrtc_streamer(key="key")
	#opencamera = st.button("Open Camera")
	#print('opncam',opencamera)
	#capture = st.button("Capture")
	#submit = st.button("Submit")
	


	#if opencamera:
	captured_image = webcam()
	print('cap',captured_image)

	if captured_image is None:

		#print("hii")
		st.write("Waiting for capture...")
		dt = datetime.now()
		# getting the timestamp
		ts = datetime.timestamp(dt)
		print("Date and time is:", dt)
		print("Timestamp is:", ts)
	else:
		captureimg_list.append(captured_image)
		dt = datetime.now()
		print("captured an image")
		# # getting the timestamp
		# ts = datetime.timestamp(dt)
		# print("Date and time is:", dt)
		# print("Timestamp is:", ts)


		if st.button("Preview Images"):
			#import pdb;pdb.set_trace()
			#for img in captureimg_list:
			print("ss", captureimg_list)
			st.image(captureimg_list)



		#st.image(captured_image)
		bio = io.BytesIO()
		captured_image.save(bio, format="PNG")
		bio.seek(0)
		#captured_image=captured_image.convert('RGB')
		#path = "captured_image/"+Username+empId+".jpg"
		#captured_image.save(path)
		if temp:
			submit = st.button("Submit")
			if submit:
				responses = (requests.post("http://localhost:5000/facerecog/home/register" , data={'name':Username, 'empId':empId} ,files = {'image':bio}))
				st.write(responses.json()['message'])


# return requests.post("http://localhost:5000/facerecog/home/register" , data={'name':Username, 'empId':empId} ,files = {'image':bio})


if __name__ =='__main__':
	print("Main progrM")
	Registrationcamera_main()
print("ourside")










	

