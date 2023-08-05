from keras_vggface.vggface import VGGFace
from src.logger import logger

MODEL = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

logger.info("VGGFace model loaded successfully.")
