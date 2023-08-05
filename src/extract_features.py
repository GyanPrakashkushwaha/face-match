from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image
import numpy as np
import cv2

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

print(model.summary())


def feature_extractor(img_path , model):
    img_array = cv2.resize(cv2.imread(filename=img_path),dsize=((224,224))) # Here I just reading the image and resizing it.
  
    expanded_img = np.expand_dims(img_array,axis=0) # Insert a new axis that will appear at the axis position in the expanded array shape.

    preprocessed_img = preprocess_input(expanded_img) #This function preprocesses the input image array according to the requirements of the specific pre-trained model

    return model.predict(preprocessed_img) # This method takes the input data, performs forward propagation through the model, and generates predictions for the input data


