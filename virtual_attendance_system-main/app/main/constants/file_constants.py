import os

#FACE_CASCADE = os.chdir(r"C:\Users\ADMIN\Desktop\facial_recognition_Arvind\app\main\static\xml_files")
#DATA_FILE = os.chdir(r"C:\Users\ADMIN\Desktop\facial_recognition_Arvind\app\main\static\data")

FACE_CASCADE = os.path.abspath('app/haarcascade_frontalface_default.xml')
DATA_FILE = os.path.abspath('userfeatures.csv')