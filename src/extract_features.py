from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image
import numpy as np
import cv2
import pickle
import tqdm
from src.exceptions import CustomException
import sys

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

imgs_file_paths = pickle.load(file=open(file=r'model/img_files_path.pkl',mode='rb'))

def feature_extractor(img_path , model):
    img_array = cv2.resize(cv2.imread(filename=img_path),dsize=((224,224))).astype(np.float32) # Here I just reading the image and resizing it.
  
    expanded_img = np.expand_dims(img_array,axis=0) # Insert a new axis that will appear at the axis position in the expanded array shape.

    preprocessed_img = preprocess_input(expanded_img) #This function preprocesses the input image array according to the requirements of the specific pre-trained model

    return model.predict(preprocessed_img) # This method takes the input data, performs forward propagation through the model, and generates predictions for the input data



imgs_features = []  # making the features list .

# print(imgs_features)
# imgs_features.clear()

try:
    for file in tqdm.tqdm(imgs_file_paths):
        imgs_features.append(feature_extractor(img_path=file,model=model))
except Exception:
    raise CustomException(Exception , sys)
    