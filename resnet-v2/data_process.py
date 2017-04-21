import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

def load_data():
    data = np.empty((42000,1,28,28),dtype="float32")
    label = np.empty((42000,),dtype="uint8")

    imgs = os.listdir("./mnist")
    num = len(imgs)
    for i in range(num):
        img = Image.open("./mnist/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        label[i] = int(imgs[i].split('.')[0])
    return label,data


