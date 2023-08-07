from src.detect_face import detect_face
import numpy as np
from keras_vggface.utils import preprocess_input
from src.constants import MODEL
from sklearn.metrics.pairwise import cosine_similarity
from src.exceptions import CustomException
import sys
from src.utils import load_pkl
import cv2
import matplotlib.pyplot as plt
from src.logger import logger

class Recommend:
    def __init__(self) -> None:
        # self.face_array = detect_face(img)
        logger.info("Face detected and initialized.") 
        pass
    
    def prediction(self,face_array1 , face_array2):
        # face_arrayss = self.face_array
        expanded_img1 = np.expand_dims(face_array1, axis=0)
        expanded_img2 = np.expand_dims(face_array2, axis=0)
        preprocessed_img1 = preprocess_input(expanded_img1)
        preprocessed_img2 = preprocess_input(expanded_img2)
        
        logger.info("Face prediction started.")
        result1 = MODEL.predict(preprocessed_img1)
        result2 = MODEL.predict(preprocessed_img2)
        logger.info("Face prediction completed.")
        return result1 , result2

    def similarity(self, result1,result2):
        sm_score = cosine_similarity(result1 ,result2)
        return sm_score

        








