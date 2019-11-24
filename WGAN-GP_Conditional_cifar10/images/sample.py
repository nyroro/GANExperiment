from __future__ import print_function, division
import keras
from keras.datasets import cifar10
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam,SGD
from functools import partial
import numpy as np

import keras.backend as K

import matplotlib.pyplot as plt
latent_dim = 100
nclasses = 10
def build_generator():
    model = Sequential()

    model.add(Dense(4*4*4*128, input_dim=latent_dim))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Reshape((4, 4, 4*128)))
    model.add(Conv2DTranspose(2*128, (2,2), strides=(2,2)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(128, (2,2), strides=(2,2)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(3, (2,2), strides=(2,2)))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(latent_dim,))

    label = Input(shape=(1,))
    label_embedding = Flatten()(Embedding(nclasses, latent_dim)(label))
    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)
(X, Y),(_,_) = cifar10.load_data()
ori_data = []
for i in range(nclasses):
    print(np.argwhere(Y==i)[0][0])
    ori_data.append(X[np.argwhere(Y==i)[0][0]])
model = build_generator()
model.load_weights('wgan_G_1.h5')
r, c = nclasses, 9
noise_sample = np.random.normal(0, 1, (r * c, latent_dim))
label_sample = np.array([[t]*c for t in range(r)]).reshape(r*c,1)
gen_imgs = model.predict([noise_sample, label_sample])

# Rescale images 0 - 1
gen_imgs = 0.5 * gen_imgs + 0.5

fig, axs = plt.subplots(r, c+1)
cnt = 0

for i in range(r):
    axs[i,0].imshow(ori_data[i])
    axs[i,0].axis('off')
    for j in range(c):
        axs[i,j+1].imshow(gen_imgs[cnt, :,:,:])
        axs[i,j+1].axis('off')
        cnt += 1
plt.savefig('cifar10.png')
plt.close()