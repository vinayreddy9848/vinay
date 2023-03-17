import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from webcam import webcam
import requests
import io
import base64
import json


def Registrationcamera_main():
    st.title("Registration")

    Username = st.text_input("USERNAME")
    print("Username", Username)

    empId = st.text_input("Employee ID")

    # st.subheader("Press start to open the camera")
    # webrtc_streamer(key="key")
    # opencamera = st.button("Open Camera")
    # print('opncam',opencamera)
    # capture = st.button("Capture")
    # submit = st.button("Submit")

    # if opencamera:
    captured_image = webcam()
    print('cap', captured_image)

    if captured_image is None:
        # print("hii")
        st.write("Waiting for capture...")

    else:
        print("hi")
        st.write("Got an image from webcamera")
        st.image(captured_image)
        print("beforeconverting",type(captured_image))

        print("type=",type(captured_image))
        bio = io.BytesIO()
        captured_image.save(bio, format="PNG")
        bio.seek(0)
        print("printing bio type",type(bio))
        captured_image = base64.b64encode(bio.read())
        print(("afterconverting",type(captured_image)))
        # captured_image=captured_image.convert('RGB')
        # path = "captured_image/"+Username+empId+".jpg"
        # captured_image.save(path)
        st.button("Submit")
        return requests.post("http://localhost:5000/facerecog/home/emp_register", json=json.dumps({'name': Username, 'empId': empId,'image': str(captured_image)}))



if __name__ == '__main__':
    Registrationcamera_main()













