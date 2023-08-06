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
    
    def prediction(self,face_array):
        # face_arrayss = self.face_array
        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        
        logger.info("Face prediction started.")
        result = MODEL.predict(preprocessed_img)
        logger.info("Face prediction completed.")
        return result

    def similarity_list(self, features_list,result):
        # self.result = self.prediction(face_array)
        # result = result
        try:
            similarity = []
            for i in range(len(features_list)):
                similarity_for_each = cosine_similarity(result, features_list[i])[0][0]
                similarity.append(similarity_for_each)

            return similarity
        except Exception as e:
            logger.error(f"Error occurred in similarity_list method: {e}")
            raise CustomException(e, sys)
        
    def recommend(self, similarity_lst):
        self.similarity = []
        self.index = []
        def feature(x):
            return x[1]
        most_similar_5_imgs = sorted(list(enumerate(similarity_lst)), reverse=True, key=feature)[0:5]

        return most_similar_5_imgs
    
    def show_similar_img(self, file_path, most_similars):
        try:
            for i in most_similars:
                img = cv2.imread(file_path[i[0]])
                plt.imshow(img)
                plt.title(label=f'similarity score: {i[1]}')
                plt.show()
        except Exception as e:
            logger.error(f"Error occurred in show_similar_img method: {e}")
            raise CustomException(e, sys)
 
                
            

        








