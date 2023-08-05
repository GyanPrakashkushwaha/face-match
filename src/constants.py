from keras_vggface.vggface import VGGFace

MODEL = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
