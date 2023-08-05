from src.utils import load_pkl , dump_pkl
import numpy as np
from src.extract_features import model
from mtcnn import MTCNN 
import cv2
from pathlib import Path
import os
import matplotlib.pyplot as plt
from PIL import Image
from keras_vggface.utils import preprocess_input

# feature_list = np.array(load_pkl(r'model\img_features.pkl'))
# file_names = load_pkl(r'model\img_files_path.pkl')

detector = MTCNN()

sample_img = cv2.imread(r'data_path\ranbir_kapoor.png')
result = detector.detect_faces(sample_img)

X , y,width,height = result[0]['box']


face =sample_img[y:y+height,X:X+width]

# plt.imshow(face)
# plt.show()
img = Image.fromarray(face)
img.resize(size=(224,224))
face_array = np.asarray(img).astype(np.float32)

expanded_img=np.expand_dims(face_array,axis=0)
preprocessed_img = preprocess_input(expanded_img)

model.predict(preprocessed_img)

