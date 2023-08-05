import streamlit as st
from src.utils import save_uploaded_img
from PIL import Image
# from src.detect_face import detect_face
# from src.recommend import Recommend
from src.utils import load_pkl

st.title('Which Celebrety is your sibling?')

img = st.file_uploader('Upload Image')

if img is not None:
    if save_uploaded_img(img):
        displayImg = Image.open(img)
        st.image(displayImg)
        
        features_list = load_pkl('model/img_features.pkl')
        st.write(features_list)
        

        




