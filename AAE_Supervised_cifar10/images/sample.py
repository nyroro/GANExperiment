import keras
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
with open('aae_decoder_tmp.json') as f:
    model = model_from_json(f.read())
model.load_weights('aae_decoder_tmp_weights.h5')
nclasses=10
latent_dim=256

r, c = nclasses, 9
noise_sample = np.random.normal(0, 1, (r * c, latent_dim))
label_sample = np.array([[t]*c for t in range(r)]).reshape(r*c,1)
label_sample = keras.utils.to_categorical(label_sample, nclasses)
gen_imgs = model.predict([noise_sample, label_sample])

from keras.datasets import cifar10
(X, Y),(_,_) = cifar10.load_data()
ori_data = []
for i in range(nclasses):
    print(np.argwhere(Y==i)[0][0])
    ori_data.append(X[np.argwhere(Y==i)[0][0]])
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