from src.detect_face import detect_face
import numpy as np
from keras_vggface.utils import preprocess_input
from src.constants import MODEL

face_arry = detect_face(r'data_path\ranbir_kapoor.png')

def recommend(face_array,model):
    expanded_img=np.expand_dims(face_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    print(model.predict(preprocessed_img))

recommend(face_array=face_arry,model=MODEL)



