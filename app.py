import streamlit as st
from src.utils import save_uploaded_img
from PIL import Image
from src.detect_face import detect_face
from src.recommend import Recommend
from src.utils import load_pkl
import os

st.set_page_config(page_title="Two-Match", page_icon=":😄:", layout="wide", initial_sidebar_state="expanded")


st.title('Which celebrity is your Sibling🤔💭')
st.markdown('>##### This model contains 8,664 photos featuring 100 distinct actors, Utilizing cosine similarity calculations, the model effectively identifies and retrieves the most akin images that bear resemblance to the inputted facial features, enabling precise face matching capabilities.')

img = st.file_uploader('Upload Image')
os.makedirs('uploaded_images',exist_ok=True)
if img is not None:
    
    if save_uploaded_img(img):
        displayImg = Image.open(img)
        st.image(image=displayImg,width=350,channels='BGR',caption='Your Image')

        st.markdown('---')

        face_arrayss = detect_face(image_path=os.path.join('uploaded_images',img.name))

        rec = Recommend()
        opt = rec.prediction(face_arrayss)
        


        # similarities = rec.similarity_list(features_list=features_list,result=opt)
        # st.write(similarity)

        # recommendation = rec.recommend(similarity_lst=similarities)
        # st.write(recommend)

        # st.subheader('Your Sibling celebrities🙈😍:')
        # st.markdown('<br/>',unsafe_allow_html=True)
