import streamlit as st
from src.utils import save_uploaded_img
from PIL import Image

st.title('Which Celebrety is your sibling?')

img = st.file_uploader('Upload Image')

if img is not None:
    if save_uploaded_img(img):
        displayImg = Image.open(img)
        st.image(displayImg)




