import streamlit as st
from src.utils import save_uploaded_img
from PIL import Image
from src.extract_features import extract_features_uploaded_img
from src.detect_face import detect_face
from src.recommend import Recommend
from src.utils import load_pkl
from src.constants import MODEL
import os



# def extract_features_uploaded_img(imgs_paths , model):
#     return feature_extractor(img_path=imgs_paths,model=model)

        

st.title('Which Celebrety is your sibling?')

img = st.file_uploader('Upload Image')

if img is not None:
    if save_uploaded_img(img):
        displayImg = Image.open(img)
        st.image(displayImg)
        
        features_list = load_pkl('model/img_features.pkl')
        # st.write(features_list)

        uploaded_img_features = extract_features_uploaded_img(imgs_paths=os.path.join('uploaded_images',img.name),model=MODEL)
        st.write(uploaded_img_features)
        print(uploaded_img_features)
        
        

        




