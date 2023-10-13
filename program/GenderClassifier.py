import numpy as np
import tensorflow as tf
import cv2
import imutils

class GenderClassifier():
    def __init__(self, model_filename='weights/model_150x200.h5'):
        self.model = tf.keras.models.load_model(model_filename)
        self.thai_alphabet_label = ['male', 'female']
                                
    def classify(self,img_org):
        image = img_org.copy()
        image = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (200,150))/255
        image = np.array([image] ,dtype=np.float32)
        
        output = self.model.predict(image)
        return (output)