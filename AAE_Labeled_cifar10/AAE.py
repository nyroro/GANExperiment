# AAE for CIFAR10
from __future__ import print_function, division
import keras
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

class AdversarialAutoencoder():
    def __init__(self):
        self.start_epochs = 0
        self.nclasses = 10
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 256

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=SGD(),
            metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        labels = Input(shape=(self.nclasses,))
        # The discriminator determines validity of the encoding
        validity = self.discriminator([encoded_repr,labels])

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model([img, labels], [reconstructed_img, validity])
        
        # self.decoder.load_weights('saved_model/aae_generator_tmp_weights.hdf5')
        # self.adversarial_autoencoder.load_weights('saved_model/aae_discriminator_tmp_weights.hdf5')


        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.99, 0.01],
            optimizer=optimizer)


    def build_encoder(self):
        # Encoder

        img = Input(self.img_shape)

        model = Sequential()
        model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=self.img_shape))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(2*128, kernel_size=3, strides=2))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(4*128, kernel_size=3, strides=2))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        h = model(img)

        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        t1 = lambda p: p[0] +K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2)
        latent_repr = Lambda(t1)([mu, log_var])

        return Model(img, latent_repr)

    def build_decoder(self):

        model = Sequential()
        model.add(Dense(4*4*4*128, input_dim=self.latent_dim))
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

        z = Input(shape=(self.latent_dim,))
        img = model(z)

        return Model(z, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim+self.nclasses))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        labels = Input(shape=(self.nclasses,))
        dis_input = Concatenate()([encoded_repr, labels])
        print(dis_input)
        validity = model(dis_input)

        return Model([encoded_repr, labels], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, Y_train), (_, _) = cifar10.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        Y_train = keras.utils.to_categorical(Y_train, self.nclasses)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            for i in range(5):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                labels = Y_train[idx]

                latent_fake = self.encoder.predict(imgs)
                labels_fake = np.random.randint(0, self.nclasses, labels.shape[0])
                labels_fake = keras.utils.to_categorical(labels_fake, self.nclasses)
                latent_real = np.random.normal(size=(batch_size, self.latent_dim))

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([latent_real, labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([latent_fake, labels_fake], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            labels_fake = np.random.randint(0, self.nclasses, labels.shape[0])
            labels_fake = keras.utils.to_categorical(labels_fake, self.nclasses)
            g_loss = self.adversarial_autoencoder.train_on_batch([imgs, labels_fake], [imgs, valid])

            # Plot the progress
            with open('result.txt', 'a') as f:
                f.write("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]\n" % (self.start_epochs+epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:

                print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (self.start_epochs+epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
                self.sample_images(epoch)
                self.save_model('tmp')

    def sample_images(self, epoch):
        epoch = self.start_epochs+epoch
        r, c = 5, 5

        z = np.random.normal(size=(r*c, self.latent_dim))
        gen_imgs = self.decoder.predict(z)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/cifar_%d.png" % epoch)
        plt.close()

    def save_model(self, name):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.h5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.decoder, "aae_decoder_%s"%name)
        save(self.encoder, "aae_encoder_%s"%name)
        save(self.discriminator, "aae_D_%s"%name)


if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=100000, batch_size=32, sample_interval=200)
    aae.save_model('cifar10')