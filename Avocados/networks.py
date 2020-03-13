from keras.layers import Conv3D, SeparableConv2D, Dense, Reshape, Dropout
from keras.layers import Input, GlobalAveragePooling2D,  AveragePooling2D, DepthwiseConv2D
from keras.layers import BatchNormalization, Activation
from keras.models import Model

import keras.backend as k

k.set_image_data_format('channels_last')
k.set_learning_phase(1)


def hyper3dnet(img_shape=(256, 256, 50, 1), classes=2):
    # Input
    d0 = Input(shape=img_shape)

    # 3D convolutions
    conv_layer1 = Conv3D(filters=1, kernel_size=(3, 3, 5), padding='same')(d0)
    conv_in = conv_layer1

    conv_in = Reshape((conv_in.shape[1].value, conv_in.shape[2].value,
                       conv_in.shape[3].value * conv_in.shape[4].value))(conv_in)

    conv_in = SeparableConv2D(96, kernel_size=5, strides=(1, 1), padding='same')(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(96, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(96, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = Activation('relu')(conv_in)

    conv_in = GlobalAveragePooling2D()(conv_in)

    if classes == 2:
        fc1 = Dense(1, name='fc' + str(1), activation='sigmoid')(conv_in)
    else:
        fc1 = Dense(classes, name='fc' + str(classes), activation='softmax')(conv_in)

    return Model(d0, fc1)


def hyper3dnet2(img_shape=(256, 256, 50, 1), classes=2):
    # Input
    d0 = Input(shape=img_shape)

    # 3D convolutions
    conv_in = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same')(d0)

    # 2D convolutions
    conv_in = Reshape((conv_in.shape[1].value, conv_in.shape[2].value,
                       conv_in.shape[3].value * conv_in.shape[4].value))(conv_in)
    conv_in = Dropout(0.05)(conv_in)

    conv_in = SeparableConv2D(320, kernel_size=5, strides=(1, 1), padding='same')(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(256, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(256, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = Activation('relu')(conv_in)

    conv_in = GlobalAveragePooling2D()(conv_in)

    if classes == 2:
        fc1 = Dense(1, name='fc' + str(1), activation='sigmoid')(conv_in)
    else:
        fc1 = Dense(classes, name='fc' + str(classes), activation='softmax')(conv_in)

    return Model(d0, fc1)


def hyper3dnet_simplified(img_shape=(256, 256, 50, 1), classes=2):
    # Input
    d0 = Input(shape=img_shape)

    conv_in = SeparableConv2D(96, kernel_size=5, strides=(1, 1), padding='same')(d0)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = AveragePooling2D(pool_size=(2, 2))(conv_in)
    conv_in = SeparableConv2D(96, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = AveragePooling2D(pool_size=(2, 2))(conv_in)
    conv_in = SeparableConv2D(96, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)

    conv_in = GlobalAveragePooling2D()(conv_in)

    if classes == 2:
        fc1 = Dense(1, name='fc' + str(1), activation='sigmoid')(conv_in)
    else:
        fc1 = Dense(classes, name='fc' + str(classes), activation='softmax')(conv_in)

    return Model(d0, fc1)


def hyper3dnet_simplified2(img_shape=(256, 256, 50, 1), classes=2):
    # Input
    d0 = Input(shape=img_shape)

    conv_in = DepthwiseConv2D(kernel_size=5, strides=(1, 1), padding='same')(d0)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = AveragePooling2D(pool_size=(2, 2))(conv_in)
    conv_in = SeparableConv2D(32, kernel_size=3, strides=(1, 1), padding='same')(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = AveragePooling2D(pool_size=(2, 2))(conv_in)
    conv_in = SeparableConv2D(64, kernel_size=3, strides=(1, 1), padding='same')(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)

    conv_in = GlobalAveragePooling2D()(conv_in)

    if classes == 2:
        fc1 = Dense(1, name='fc' + str(1), activation='sigmoid')(conv_in)
    else:
        fc1 = Dense(classes, name='fc' + str(classes), activation='softmax')(conv_in)

    return Model(d0, fc1)
