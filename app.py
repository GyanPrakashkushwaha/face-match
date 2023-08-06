import streamlit as st
from src.utils import save_uploaded_img
from PIL import Image
from src.detect_face import detect_face
from src.recommend import Recommend
from src.utils import load_pkl
import os


st.title('Which Celebrety is your sibling?')

img = st.file_uploader('Upload Image')
os.makedirs('uploaded_images',exist_ok=True)
if img is not None:
    
    if save_uploaded_img(img):
        displayImg = Image.open(img)
        st.image(image=displayImg,width=350,channels='BGR',caption='Your Image')
        # st.write(cv2.imread(os.path.join('uploaded_images',img.name)))


        features_list = load_pkl('model/img_features.pkl')
        file_path = load_pkl('model/img_files_path.pkl')
        face_arrayss = detect_face(image_path=os.path.join('uploaded_images',img.name))
        # st.write(face_arrayss)

        rec = Recommend()
        opt = rec.prediction(face_arrayss)

        # st.write(opt)
        # st.write(type(opt.reshape(1,-1)))


        similarities = rec.similarity_list(features_list=features_list,result=opt)
        # st.write(similarity)

        recommendation = rec.recommend(similarity_lst=similarities)
        # st.write(recommend)

        st.subheader('Your Sibling Celebreties:')

        imgs = []
        similarity_score = []
        actor_name = []
        for i in recommendation:
            imgs.append(file_path[i[0]])
            similarity_score.append(i[1])
            parts = file_path[i[0]].split('\\')[1].split('_')
            actor_name.append(' '.join(parts))

        st.write(imgs)
        st.write(similarity_score)
        st.write(actor_name)


        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"<p style='text-align: center;'>{actor_name[0]}\n"
                        f"similarity_score: {similarity_score[0]}  \n"
                                    , unsafe_allow_html=True)
            st.image(image=displayimgs[0],width=350,channels='BGR')

        with col2:
            st.markdown(f"<p style='text-align: center;'>{actor_name[1]}\n"
                        f"similarity_score: {similarity_score[1]}  \n"
                                    , unsafe_allow_html=True)
            st.image(image=displayimgs[1],width=350,channels='BGR')

        with col3:
            st.markdown(f"<p style='text-align: center;'>{actor_name[2]}\n"
                        f"similarity_score: {similarity_score[2]}  \n"
                                    , unsafe_allow_html=True)
            st.image(image=displayimgs[2],width=350,channels='BGR')

        with col4:
            st.markdown(f"<p style='text-align: center;'>{actor_name[3]}\n"
                        f"similarity_score: {similarity_score[3]}  \n"
                                    , unsafe_allow_html=True)
            st.image(image=displayimgs[3],width=350,channels='BGR')

        with col5:
            st.markdown(f"<p style='text-align: center;'>{actor_name[4]}\n"
                        f"similarity_score: {similarity_score[4]}  \n"
                                    , unsafe_allow_html=True)
            st.image(image=displayimgs[4],width=350,channels='BGR')












         

        




