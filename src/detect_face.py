import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2
import os
from PIL import Image
from src.exceptions import CustomException
import sys
from src.logger import logger



def detect_face(image_path):
    detector = MTCNN()
    os.chdir(r'd:\\vscode_machineLearning\\BEST_PROJECTS\\face-match')
    try:
        logger.info(f"Loading image from {image_path}.")
        sample_img = cv2.imread(image_path)
        logger.info("Detecting face in the image.")
        detected_face = detector.detect_faces(sample_img)
        
        if len(detected_face) == 0:
            raise CustomException("No face detected in the image.", sys)
        
        X, y, width, height = detected_face[0]['box']
        face = sample_img[y:y+height, X:X+width]
        img = Image.fromarray(face)
        img = img.resize(size=(224, 224))
        face_array = np.asarray(img).astype(np.float32)

        return face_array
    except Exception as e:
        logger.error(f"Error occurred in detect_face method: {e}")
        print(e)
