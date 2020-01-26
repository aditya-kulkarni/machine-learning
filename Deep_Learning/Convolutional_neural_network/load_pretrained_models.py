import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50
 
#Loads the VGG16 model
vgg_model = vgg16.VGG16(weights='imagenet')
 
# Loads the Inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')
 
# Loads the ResNet50 model 
resnet_model = resnet50.ResNet50(weights='imagenet')
