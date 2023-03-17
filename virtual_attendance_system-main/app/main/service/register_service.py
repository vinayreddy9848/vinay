import os
import sys

import cv2
import face_recognition
from app.main.constants.file_constants import FACE_CASCADE, DATA_FILE
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import keras
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from io import BytesIO
from PIL import Image
import time
from datetime import datetime
import matplotlib.pyplot as plt
#global model

def load_vgg_resnet():
    # model = VGGFace(include_top=False,pooling='avg') # default : VGG16 , you can use model='resnet50' or 'senet50'
    return VGGFace(model='senet50', include_top=False, pooling='avg')


def load_models():
    model = load_vgg_resnet()
    return model

def gen(camera):
    while True:
        frame = camera.get_frame()
        YIELD= yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        print("YIELD",YIELD)



class VideoCamera(object):
    def __init__(self, image=None):
        self.video = cv2.VideoCapture(0)
        #self.video.set(cv2.CAP_PROP_FPS0, 6)
        #print("frame=", self.video.get(cv2.CAP_PROP_FPS))
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
        self.csv_file = DATA_FILE
        self.image = image

    def __del__(self):
        self.video.release()
    #
    # path = 'app\main\static'
    # images = []
    #
    # classNames = []
    # myList = os.listdir(path)
    # print("MYLIST=", myList)
    # for cl in myList:
    #     curImg = cv2.imread(f'{path}\{cl}')
    #     #curImg = plt.imread(f'{path}/{cl}')
    #     print("CurImg=", curImg)
    #     images.append(curImg)
    #
    #     classNames.append(os.path.splitext(cl)[0])
    # print(classNames)

    def markattendance(self, name):
        file = open("Attandance.txt", "a")
        file.write(name+"\n")
        file.close()
        pass

    def resize_img(self, img, width, height):
        dim = (width, height)
        print("in resizze",type(img))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        #resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized

    def detect_crop_face_hog(self, img_file):
        img_file = np.asarray(img_file)

        print(type(img_file))
        image = face_recognition.load_image_file(img_file)

        #     image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  # convert to gray
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     faces = face_recognition.face_locations(image,number_of_times_to_upsample=1, model='cnn')
        faces = face_recognition.face_locations(image, number_of_times_to_upsample=1, model='hog')
        if faces == []:
            #         print(image_file_name, 'Face not detected ')
            c_img = 'NOF'
            pass
        else:
            for x, y, w, h in faces:
                #     cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
                c_img = image[x:w, h:y]
        print("cimg=",c_img)
        return c_img

    def extract_features(self, img_file, p_name=None):
        # print(type(img_file))
        c_img_file = self.detect_crop_face_hog(img_file)
        print("cropped")
        r_img_file = self.resize_img(c_img_file, 224, 224)
        #print(type(r_img_file))
        print("resized")
        new_img = Image.fromarray(r_img_file)
        new_img.save("Current_img.png")
        #byte_io = BytesIO(r_img_file.tobytes())
        #print("array_from_bytes", array_from_bytes)
        img = keras.utils.load_img("Current_img.png", target_size=(224, 224))
        x = keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=2)  # or version=2
        #array_from_bytes = np.frombuffer(byte_io.getbuffer(), dtype=x).reshape(x.shape)
        return x


    def transorm(self, image):
        encoded = self.extract_features(image)
        model = load_models()
        pic1 = model.predict(encoded)
        temp_df = pd.DataFrame(pic1.ravel()).T
        return temp_df

    def identify_faces(self, df, transformed_data):
        thresh = 0.65
        file_name = 'NO MATCH FOUND'
        max_match = 0
        for rec in np.arange(0, len(df)):
            #print((df.iloc[rec,1:]))
            #print("rec=", rec)
            pic2 = np.array(df.iloc[rec, 1:]).reshape((1, 2048))
            match = cosine_similarity(transformed_data, pic2)
            if match > thresh:
                if match > max_match:
                    max_match = match
                    file_name = df.index[rec]
        return file_name, max_match

    def get_frame(self):
        success, raw_image = self.video.read()
        image = cv2.resize(raw_image, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        df = pd.read_csv(self.csv_file)

        for (x, y, w, h) in face_rects:
            transformed_data = self.transorm(raw_image)
            match_found = self.identify_faces(df, transformed_data)
            print('Match Found = ',match_found)
            if match_found != "NO MATCH FOUND":
                name = df.iloc[match_found[0],0]
                print("name=", name)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x + 6, (y + h) - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                self.markattendance(name)
                break
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def r1(self,name,empId):
        print("register")
        image = self.image
        transformed_data = self.transorm(image)
        # Write a code to insert this transformed_data to user_features.csv
        transformed_data.insert(0, 'Name', name)
        print(transformed_data)
        transformed_data.to_csv('app/main/static/data/userfeatures.csv', mode='a', index=False, header=False)
        return("success")

    def register(self, name, empId):
        try:
            print("register")
            image = self.image
            transformed_data = self.transorm(image)
            # Write a code to insert this transformed_data to user_features.csv
            transformed_data.insert(0, 'Name', name)
            print(transformed_data)
            transformed_data.to_csv('app/main/static/data/userfeatures.csv', mode='a', index=False, header=False)

            return {"message": "Employee registered Successfully", "status": "Success", "statusCode": 200}

        except:
            print(sys.exc_info()[1])
            return {"message": "Employee registration Failed", "status": "Failure", "statusCode": 401}



def render_video():
    return gen(VideoCamera())


def register_employee(name, empId, image):
    print("regemp=")
    return VideoCamera(image).r1(name, empId)
