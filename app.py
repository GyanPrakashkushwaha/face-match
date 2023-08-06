import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import save_uploaded_img
from PIL import Image
from src.detect_face import detect_face
from src.recommend import Recommend
from src.utils import load_pkl
from src.constants import MODEL
import os
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


        features_list = load_pkl('model/img_features.pkl')
        face_arrayss = detect_face(image_path=os.path.join('uploaded_images',img.name))
        st.write(face_arrayss)

        rec = Recommend()
        opt = rec.prediction(face_arrayss)

        st.write(opt)
        st.write(type(opt.reshape(1,-1)))


        similarity = rec.similarity_list(features_list=features_list,result=opt)
        st.write(similarity)
        


         

        




