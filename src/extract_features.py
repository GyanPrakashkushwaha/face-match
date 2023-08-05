from keras_vggface.utils import preprocess_input
import numpy as np
import cv2
import tqdm
from src.exceptions import CustomException
import sys
import warnings
warnings.filterwarnings('ignore')
from src.logger import logger



def feature_extractor(img_path , model):
    img_array = cv2.resize(cv2.imread(filename=img_path),dsize=((224,224))).astype(np.float32) # Here I just reading the image and resizing it.
    # logger.info(f'converted the pictures to images')

    expanded_img = np.expand_dims(img_array,axis=0) # Insert a new axis that will appear at the axis position in the expanded array shape.

    preprocessed_img = preprocess_input(expanded_img) #This function preprocesses the input image array according to the requirements of the specific pre-trained model
    # logger.info(f'Preprocessing using [preprocess_input] method has done.')

    return model.predict(preprocessed_img).flatten() ,  # This method takes the input data, performs forward propagation through the model, and generates predictions for the input data


def extract_features(imgs_file_paths , model):
    imgs_features = []  # making the features list .

    try:
        for file in tqdm.tqdm(imgs_file_paths):
            imgs_features.append(feature_extractor(img_path=file,model=model))

        logger.info(f'appended all the imgs_features')
        return imgs_features
    
    except Exception:
        raise CustomException(Exception , sys)
    

def extract_features_uploaded_img(imgs_file_paths , model):
    return feature_extractor(img_path=imgs_file_paths,model=model)

        


    





    