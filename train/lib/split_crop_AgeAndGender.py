import os
import numpy as np
import shutil
import cv2
from FaceDetector import FaceDetector

if __name__ == '__main__':
     face_detector = FaceDetector()
     
     # set parameters
     data_path = "../data/InthewildFaces"
     save_path = "../data/New_InTheWildFaces"
     min_age = 20
     max_age = 60
     
     if not(os.path.isdir(save_path)) :
          os.mkdir(save_path)
     
     # parts
     parts = os.listdir(data_path)
     
     # process each class แยกโฟลเดอร์ตามเพศ
     for ic in range(len(parts)):
          
          # list filenames
          filenames = np.array(os.listdir(f"{data_path}/{parts[ic]}"))
          
          for i in range(len(filenames)):
               gender = filenames[i].split("_")[1]
               age = filenames[i].split("_")[0]
               des_path = save_path + "/" + gender + "/" + age
               source_filename = data_path + "/" + parts[ic] + "/" + filenames[i]
               des_filename = des_path + "/" + filenames[i]
               
               if int(age) >= min_age and int(age) <= max_age and gender != "3" and filenames[i] != "53__0_20170116184028385.jpg":
                    if not(os.path.isdir(save_path + "/" + gender) ) :
                         os.mkdir(save_path + "/" + gender)
                    if not(os.path.isdir(des_path)) :
                         os.mkdir(des_path)
                    
                    print(f"\rcopy {source_filename} to {des_filename} ... {i+1}/{len(filenames)}", end=" "*10)
                    
                    # crop 
                    image = cv2.imread(source_filename)
                    face_output = face_detector.getFace(image)
                    if len(face_output) > 0 and gender != "3" and filenames[i] != "53__0_20170116184028385.jpg":
                         cv2.imwrite(des_filename, face_output[0][2])
   