from FaceDetector import FaceDetector
from GenderClassifier import GenderClassifier
import cv2
import numpy as np
import torch

class imgInformation():
     def __init__(self):
          self.face_box = []
          self.face_img = []
     
     def putText(self, img, text, x, y, dy=30, color=(0 ,0 ,255)):
          lines = text.split("\n")
          for i in range(len(lines)):
               cv2.putText(img, lines[i], (x ,y+i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
          
     def predict(self, frame):
          face_detector = FaceDetector()
          gender_clssifier = GenderClassifier()
          
          # face_img , face_box
          faces = face_detector.getFace(frame)
          for i in range(len(faces)):
               self.face_img.append(faces[i][2])
               self.face_box.append(faces[i][0])
          
          # classify face gender and write
          for i in range(len(self.face_box)):
               gender_prob = gender_clssifier.classify(self.face_img[i])[0]
               print(gender_prob)
               
               x, y, w, h = self.face_box[i]
               text_gender_information = f"Male {gender_prob[0]*100:.2f}%" if gender_prob[0] > gender_prob[1] else f"Female {gender_prob[1]*100:.2f}%"             
               cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
               self.putText(frame, text_gender_information, x, y, dy=15)
               
if __name__ == '__main__':
     
     cap = cv2.VideoCapture(0)
     
     while(True):
          ret, frame = cap.read()
          
          img_information = imgInformation()
          
          # predict
          img_information.predict(frame)
          
          cv2.imshow('Real-Time', frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
               break
          
          img = frame
          torch.cuda.empty_cache()
          
     cap.release()
     cv2.destroyAllWindows()
     
   
          
     


     
         
         

  
     
 