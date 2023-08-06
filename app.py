import streamlit as st
from src.utils import save_uploaded_img
from PIL import Image
from src.extract_features import extract_features_uploaded_img_face_array
from src.detect_face import detect_face
from src.recommend import Recommend
from src.utils import load_pkl
from src.constants import MODEL
import os
from src.recommend import Recommend
import cv2
from keras_vggface.utils import preprocess_input
import numpy as np
from src.logger import logger

# def extract_features_uploaded_img(imgs_paths , model):
#     return feature_extractor(img_path=imgs_paths,model=model)

        

st.title('Which Celebrety is your sibling?')

img = st.file_uploader('Upload Image')
os.makedirs('uploaded_images',exist_ok=True)
if img is not None:
    
    if save_uploaded_img(img):
        displayImg = Image.open(img)
        st.image(displayImg)
        # st.write(cv2.imread(os.path.join('uploaded_images',img.name)))

        def prediction(model,face_array):
        # face_arrayss = self.face_array
            expanded_img = np.expand_dims(face_array, axis=0)
            preprocessed_img = preprocess_input(expanded_img)
            
            logger.info("Face prediction started.")
            result = model.predict(preprocessed_img)
            logger.info("Face prediction completed.")
            return result
        
        features_list = load_pkl('model/img_features.pkl')
        # st.write(features_list)
        face_arrayss = detect_face(image_path=os.path.join('uploaded_images',img.name))
        st.write(face_arrayss)

        print(prediction(face_arrayss))
        st.write(prediction(MODEL,face_arrayss))

        # uploaded_img_featuress = extract_features_uploaded_img_face_array(face_array_new=face_arrayss,model=MODEL)
        # st.write(uploaded_img_featuress)

        # rec = Recommend()
        # similarity_list_ = rec.similarity_list(features_list=features_list,face_array=face_arrayss)

        # st.write(similarity_list_)

         

        




