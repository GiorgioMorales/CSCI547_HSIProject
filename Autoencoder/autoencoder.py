from keras.layers import Dense, Reshape, Flatten, Conv3D, Add
from keras.layers import Input, MaxPooling3D, UpSampling3D
from keras.layers import BatchNormalization, Activation
from keras.models import Model

import keras.backend as k

k.set_image_data_format('channels_last')


def autoencoder2d(img_shape=(28, 28, 224, 1)):

    input_img = Input(img_shape)

    x = Conv3D(16, (3, 3, 7), activation='relu', padding='same')(input_img)
    x = MaxPooling3D((1, 1, 2), padding='same')(x)
    x = Conv3D(8, (3, 3, 7), activation='relu', padding='same')(x)
    x = MaxPooling3D((1, 1, 2), padding='same')(x)
    x = Conv3D(8, (3, 3, 7), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((1, 1, 2), padding='same')(x)

    # Decoder

    x = Conv3D(8, (3, 3, 7), activation='relu', padding='same')(encoded)
    x = UpSampling3D((1, 1, 2))(x)
    x = Conv3D(8, (3, 3, 7), activation='relu', padding='same')(x)
    x = UpSampling3D((1, 1, 2))(x)
    x = Conv3D(16, (3, 3, 7), activation='relu', padding='same')(x)
    x = UpSampling3D((1, 1, 2))(x)
    decoded = Conv3D(1, (3, 3, 7), activation='sigmoid', padding='same')(x)

    return Model(input_img, decoded)


autoencoder = autoencoder2d()
autoencoder.summary()
