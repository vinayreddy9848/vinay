import os
import sys
from flask import session
import cv2
import face_recognition
from app.main.constants.file_constants import FACE_CASCADE, DATA_FILE
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import keras
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from io import BytesIO
from PIL import Image
import time
from datetime import datetime
import matplotlib.pyplot as plt

from app.main.model.employee import Employee, Attendance



# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         YIELD = yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         print("YIELD", YIELD)


class VideoCamera(object):
    def __init__(self, image=None):
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
        self.csv_file = DATA_FILE
        self.image = image

    def markattendance(self, name):
        file = open("Attandance.txt", "a")
        file.write(name + "\n")
        file.close()
        pass

    def resize_img(self, img, width, height):
        dim = (width, height)
        # print(type(img))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        dt = datetime.now()
        # getting the timestamp
        ts = datetime.timestamp(dt)
        print("Date and time is:", dt)
        print("Timestamp is:", ts)

        # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized

    def detect_crop_face_hog(self, img_file):
        # img_file = np.asarray(img_file)
        # print(type(img_file))
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

        dt = datetime.now()
        # getting the timestamp
        ts = datetime.timestamp(dt)
        print("Date and time is:", dt)
        print("Timestamp is:", ts)
        return c_img

    def extract_features(self, img_file, p_name=None):
        # print(type(img_file))
        c_img_file = self.detect_crop_face_hog(img_file)
        print("cropped")
        r_img_file = self.resize_img(c_img_file, 224, 224)
        # print(type(r_img_file))
        print("resized")
        new_img = Image.fromarray(r_img_file)
        new_img.save("Current_img.png")
        # byte_io = BytesIO(r_img_file.tobytes())
        # print("array_from_bytes", array_from_bytes)
        img = load_img(os.path.abspath("Current_img.png"), target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=2)  # or version=2
        # array_from_bytes = np.frombuffer(byte_io.getbuffer(), dtype=x).reshape(x.shape)
        dt = datetime.now()
        # getting the timestamp
        ts = datetime.timestamp(dt)
        print("Date and time is:", dt)
        print("Timestamp is:", ts)
        return x

    def transorm(self, image, apiModel):
        encoded = self.extract_features(image)
        model = apiModel
        pic1 = model.predict(encoded)
        temp_df = pd.DataFrame(pic1.ravel()).T
        print(temp_df)
        return temp_df

    def identify_faces(self, df, transformed_data):
        thresh = 0.65
        file_name = 'NO MATCH FOUND'
        max_match = 0
        for rec in np.arange(0, len(df)):
            pic2 = np.array(df.iloc[rec, 1:]).reshape((1, 2048))
            match = cosine_similarity(transformed_data, pic2)
            if match > thresh:
                if match > max_match:
                    max_match = match
                    file_name = df.index[rec]
        return file_name, max_match

    def face_match(self, df, transformed_data):
        thresh = 0.65
        max_match = 0
        for rec in np.arange(0, len(df)):
            pic2 = np.array(df.iloc[rec, 0:]).reshape((1, 2048))
            match = cosine_similarity(transformed_data, pic2)
            if match > thresh:
                if match > max_match:
                    max_match = match
        print(max_match)
        return float(max_match[0][0]) if isinstance(max_match, list) else float(max_match)

    def get_frame(self, apiModel):
        success, raw_image = self.video.read()
        image = cv2.resize(raw_image, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        df = pd.read_csv(self.csv_file)

        for (x, y, w, h) in face_rects:
            transformed_data = self.transorm(raw_image, apiModel)
            match_found = self.identify_faces(df, transformed_data)
            print('Match Found = ', match_found)
            if match_found != "NO MATCH FOUND":
                name = df.iloc[match_found[0], 0]
                print("name=", name)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x + 6, (y + h) - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                self.markattendance(name)
                break
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def register(self, frameCount, empId, apiModel):
        try:
            image = self.image
            transformed_data = self.transorm(image, apiModel).values.tolist()[0]
            emp = Employee.query.filter_by(empId=int(empId)).first()
            features = []
            empObj = Employee()
            if emp.features is None:
                features.append(emp.features)
            else:
                features = emp.features.append(transformed_data)
            empObj.update(features)
            return {"message": "Employee Face registered Successfully", "status": "Success", "statusCode": 200}

        except:
            print(sys.exc_info())
            return {"message": "Employee registration Failed", "status": "Failure", "statusCode": 401}

    def mark(self, punch_type, face_data, apiModel):
        try:
            image = self.image
            transformed_data = self.transorm(image, apiModel)
            try:
                employees = session["employees"]
                if not employees:
                    employees = Employee.query.all()
                    session["employees"] = employees
            except:
                employees = Employee.query.all()
            max_emp = {"empName": '', "empId": None, "match": 0}
            crnt_time = datetime.now()
            for emp in employees:
                if len(emp.features) > 0:
                    for feature in emp.features:
                        df = pd.DataFrame(feature).T
                        match_found = self.face_match(df, transformed_data)
                        if match_found > max_emp["match"]:
                            max_emp["match"] = match_found
                            max_emp["empName"] = emp.empName
                            max_emp["empId"] = emp.empId
            attendance = Attendance()
            if max_emp["match"] == 0:
                val = attendance.save(0000, crnt_time, punch_type, False, face_data)
                return {"message": "No Matching Employee found", "status": "Failure", "statusCode": 401}
            val = attendance.save(max_emp["empId"], crnt_time, punch_type, False, None)
            return {"data": {"id": val, "empName": max_emp["empName"], "empId": max_emp["empId"]},
                    "message": "Confirm Whether its' you", "status": "Success", "statusCode": 200}
        except:
            print(sys.exc_info())
            return {"message": "Attendance Failed!", "status": "Failure", "statusCode": 500}

    def confirm(self, id, confirm_value):
        try:
            atd = Attendance.query.get(id)
            emp = Employee.query.filter_by(empId=atd.empId).first()
            atd.confirm = confirm_value
            Attendance.update()
            return {"data": {"id": atd.id, "empName": emp.empName, "empId": atd.empId},
                    "message": "Attendance Marked", "status": "Success", "statusCode": 200}
        except:
            print(sys.exc_info())
            return {"message": "Attendance Failed!", "status": "Failure", "statusCode": 500}

    def attendance_list(self):
        try:
            response = []
            employees = Attendance.query.all()
            for emp in employees:
                response.append({"empId": emp.empId, "time": emp.time.strftime("%m/%d/%Y, %H:%M"),
                                 "punch": emp.punch, "confirmed": emp.confirmed})
            return {"data": response, "message": "Attendance listed", "status": "Success", "statusCode": 200}
        except:
            print(sys.exc_info())
            return {"message": "Attendance Failed!", "status": "Failure", "statusCode": 500}



def register_employee(frameCount, empId, image, apiModel):
    return VideoCamera(image).register(frameCount, empId, apiModel)


def make_attendance(image, punch_type, face_data, apiModel):
    return VideoCamera(image).mark(punch_type, face_data, apiModel)


def confirm_attendance(id, confirm):
    return VideoCamera(None).confirm(id, confirm)


def display_attendance():
    return VideoCamera(None).attendance_list()