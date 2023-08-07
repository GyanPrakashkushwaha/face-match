import streamlit as st
from src.utils import save_uploaded_img
from PIL import Image
from src.detect_face import detect_face
from src.recommend import Recommend
from src.utils import load_pkl
import os

st.set_page_config(page_title="Two-Match", page_icon=":😄:", layout="wide", initial_sidebar_state="expanded")

col1 ,col2 = st.columns(2)

with col1:
    img1 = st.file_uploader('Upload Image')

with col2:
    img2 = st.file_uploader('Upload New Image')





# os.makedirs('uploaded_images',exist_ok=True)
# if img is not None:
    
#     if save_uploaded_img(img):
#         displayImg = Image.open(img)
#         st.image(image=displayImg,width=350,channels='BGR',caption='Your Image')

#         st.markdown('---')

#         face_arrayss = detect_face(image_path=os.path.join('uploaded_images',img.name))

#         rec = Recommend()
#         opt = rec.prediction(face_arrayss)
        


#         # similarities = rec.similarity_list(features_list=features_list,result=opt)
#         # st.write(similarity)

        # recommendation = rec.recommend(similarity_lst=similarities)
        # st.write(recommend)

        # st.subheader('Your Sibling celebrities🙈😍:')
        # st.markdown('<br/>',unsafe_allow_html=True)
