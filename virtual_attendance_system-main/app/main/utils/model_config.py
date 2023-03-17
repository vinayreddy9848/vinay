from keras_vggface.vggface import VGGFace
from keras_vggface import utils

global model

def load_vgg_resnet():
    # model = VGGFace(include_top=False,pooling='avg') # default : VGG16 , you can use model='resnet50' or 'senet50'
    return VGGFace(model='senet50', include_top=False, pooling='avg')


def load_models():
    model = load_vgg_resnet()
    return
