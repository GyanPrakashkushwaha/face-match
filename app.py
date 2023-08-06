import streamlit as st
from src.utils import save_uploaded_img
from PIL import Image
from src.detect_face import detect_face
from src.recommend import Recommend
from src.utils import load_pkl
import os

st.set_page_config(page_title="Face-Match", page_icon=":😄:", layout="wide", initial_sidebar_state="expanded")


st.title('Which Celebrety is your Sibling🤔💭')
st.markdown('>##### This model contains 8,664 photos featuring 100 distinct actors, Utilizing cosine similarity calculations, the model effectively identifies and retrieves the most akin images that bear resemblance to the inputted facial features, enabling precise face matching capabilities.')

img = st.file_uploader('Upload Image')
os.makedirs('uploaded_images',exist_ok=True)
if img is not None:
    
    if save_uploaded_img(img):
        displayImg = Image.open(img)
        st.image(image=displayImg,width=350,channels='BGR',caption='Your Image')

        st.markdown('---')


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

        st.subheader('Your Sibling Celebreties🙈😍:')

        imgs = []
        similarity_score = []
        actor_name = []
        for i in recommendation:
            imgs.append(file_path[i[0]])
            similarity_score.append(i[1])
            parts = file_path[i[0]].split('\\')[1].split('_')
            actor_name.append(' '.join(parts))

        similarity_score_lst = [round(i*100,ndigits=2) for i in similarity_score]
        col1, col2, col3, col4, col5 = st.columns(5)

        # Add custom CSS to style the webpage
        st.markdown(
            """
            <style>
                body {
                    background-color: #f0f2f6; /* Set the background color to a light gray */
                    color: #333; /* Set the text color to dark gray */
                    font-family: 'Arial', sans-serif; /* Change the font-family to Arial or any other preferred font */
                }
                .stButton {
                    background-color: #3498db; /* Set the button background color to a blue shade */
                    color: white; /* Set the button text color to white */
                    font-weight: bold; /* Make the button text bold */
                }
                .stFileUploader label {
                    color: #3498db; /* Set the file uploader label text color to blue */
                }
                .stFileUploader small {
                    color: #666; /* Set the file uploader small text color to dark gray */
                }
                .stImage {
                    border: 3px solid #3498db; /* Add a blue border around the images */
                    border-radius: 5px; /* Add a slight border-radius to the images */
                }
                .stMarkdown {
                    line-height: 1.6; /* Increase the line height for better readability */
                }
            </style>
            """,
            unsafe_allow_html=True
        )



                
        with col1:
            st.markdown(f"<p style='text-align: center;'><b>{actor_name[0]}</b><br>"
                        f"<b>Similarity Score:</b> {similarity_score_lst[0]}%</p>", unsafe_allow_html=True)
            st.image(image=imgs[0], width=200, channels='BGR')

        with col2:
            st.markdown(f"<p style='text-align: center;'><b>{actor_name[1]}</b><br>"
                        f"<b>Similarity Score:</b> {similarity_score_lst[1]}%</p>", unsafe_allow_html=True)
            st.image(image=imgs[1], width=200, channels='BGR')

        with col3:
            st.markdown(f"<p style='text-align: center;'><b>{actor_name[2]}</b><br>"
                        f"<b>Similarity Score:</b> {similarity_score_lst[2]}%</p>", unsafe_allow_html=True)
            st.image(image=imgs[2], width=200, channels='BGR')

        with col4:
            st.markdown(f"<p style='text-align: center;'><b>{actor_name[3]}</b><br>"
                        f"<b>Similarity Score:</b> {similarity_score_lst[3]}%</p>", unsafe_allow_html=True)
            st.image(image=imgs[3], width=200, channels='BGR')

        with col5:
            st.markdown(f"<p style='text-align: center;'><b>{actor_name[4]}</b><br>"
                        f"<b>Similarity Score:</b> {similarity_score_lst[4]}%</p>", unsafe_allow_html=True)
            st.image(image=imgs[4], width=200, channels='BGR')

        if actor_name[1] == actor_name[2] == actor_name[3] == actor_name[4] == actor_name[0]:
            st.markdown(f'> ### {actor_name[0]} is only Yours Sibling🤭 \n made by 👨🏻‍💻Gyan Prakash Kushwaha')
        else:
            st.markdown(f"> ## Multiple actors are your sibling🤯\n made by 👨🏻‍💻Gyan Prakash Kushwaha")












         

        




