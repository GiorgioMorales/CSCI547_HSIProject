from keras.layers import Conv3D, SeparableConv2D, Dense, Reshape, Flatten, ConvLSTM2D, Permute, Dropout, Conv2D, Add
from keras.layers import Input, GlobalAveragePooling2D, AveragePooling3D, Lambda, Concatenate, Multiply, MaxPooling2D, DepthwiseConv2D
from keras.layers import BatchNormalization, Activation
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras import regularizers

import keras.backend as k

k.set_image_data_format('channels_last')
k.set_learning_phase(1)


def hyper3dnet(img_shape=(256, 256, 50, 1), classes=2):
    # Input
    d0 = Input(shape=img_shape)

    # Initial attention
    # d0 = attention_vector(d0)

    # 3D convolutions
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same')(d0)
    conv_layer1 = BatchNormalization()(conv_layer1)
    conv_layer1 = Activation('relu')(conv_layer1)
    conv_layer2 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same')(conv_layer1)
    conv_layer2 = BatchNormalization()(conv_layer2)
    conv_layer2 = Activation('relu')(conv_layer2)
    conv_in = Concatenate()([conv_layer1, conv_layer2])
    conv_layer3 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same')(conv_in)
    conv_layer3 = BatchNormalization()(conv_layer3)
    conv_layer3 = Activation('relu')(conv_layer3)
    conv_in = Concatenate()([conv_in, conv_layer3])
    #conv_layer4 = Conv3D(filters=8, kernel_size=(3, 3, 7), padding='same')(conv_in)
    #conv_layer4 = BatchNormalization()(conv_layer4)
    #conv_layer4 = Activation('relu')(conv_layer4)
    #conv_in = Concatenate()([conv_in, conv_layer4])

    # conv_in = Lambda(lambda x: k.mean(x, axis=-1))(conv_in)
    conv_in = Reshape((conv_in.shape[1].value, conv_in.shape[2].value,
                       conv_in.shape[3].value * conv_in.shape[4].value))(conv_in)
    conv_in = Dropout(0.5)(conv_in)

    conv_in = SeparableConv2D(128, kernel_size=3, strides=(1, 1), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(128, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(128, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(128, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)

    # conv_in = GlobalAveragePooling2D()(conv_in)

    conv_in = Flatten()(conv_in)
    conv_in = Dropout(0.2)(conv_in)

    if classes == 2:
        fc1 = Dense(1, name='fc' + str(1), activation='sigmoid')(conv_in)
    else:
        fc1 = Dense(classes, name='fc' + str(classes), activation='softmax')(conv_in)

    return Model(d0, fc1)
        
    
def hyper3dnet2(img_shape=(256, 256, 50, 1), classes=2):
    # Input
    d0 = Input(shape=img_shape)

    # Initial attention
    # d0 = attention_vector(d0)

    # 3D convolutions
    conv_layer1 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same')(d0)
    conv_layer1 = BatchNormalization()(conv_layer1)
    conv_layer1 = Activation('relu')(conv_layer1)  
    #conv_layer1 = AveragePooling3D(pool_size=(2, 2, 2))(conv_layer1)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same')(conv_layer1)
    conv_layer2 = BatchNormalization()(conv_layer2)
    conv_layer2 = Activation('relu')(conv_layer2)
    #conv_in = Concatenate()([conv_layer1, conv_layer2])
    conv_in = conv_layer2
    #conv_layer3 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same')(conv_in)
    #conv_layer3 = BatchNormalization()(conv_layer3)
    #conv_layer3 = Activation('relu')(conv_layer3)
    #conv_in = Concatenate()([conv_in, conv_layer3])

    # conv_in = Lambda(lambda x: k.mean(x, axis=-1))(conv_in)
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

    #conv_in = Flatten()(conv_in)
    #conv_in = Dropout(0.2)(conv_in)

    if classes == 2:
        fc1 = Dense(1, name='fc' + str(1), activation='sigmoid')(conv_in)
    else:
        fc1 = Dense(classes, name='fc' + str(classes), activation='softmax')(conv_in)

    return Model(d0, fc1)
        


def hyper3dnet_simplified(img_shape=(256, 256, 50, 1), classes=2):
    # Input
    d0 = Input(shape=img_shape)

    conv_in = SeparableConv2D(128, kernel_size=5, strides=(1, 1), padding='same')(d0)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(128, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(128, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
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

    conv_in = SeparableConv2D(128, kernel_size=5, strides=(1, 1), padding='same')(d0)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(256, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = BatchNormalization()(conv_in)
    conv_in = Activation('relu')(conv_in)
    conv_in = SeparableConv2D(256, kernel_size=3, strides=(2, 2), padding='same',
                              dilation_rate=1)(conv_in)
    conv_in = Activation('relu')(conv_in)
    #conv_in = BatchNormalization()(conv_in)

    conv_in = GlobalAveragePooling2D()(conv_in)

    #conv_in = Flatten()(conv_in)
    #conv_in = Dense(128)(conv_in)
    #conv_in = BatchNormalization()(conv_in)
    #conv_in = Activation('relu')(conv_in)
    #conv_in = Dense(128)(conv_in)
    #conv_in = BatchNormalization()(conv_in)
    #conv_in = Activation('relu')(conv_in)
    #conv_in = Dropout(0.2)(conv_in)

    if classes == 2:
        fc1 = Dense(1, name='fc' + str(1), activation='sigmoid')(conv_in)
    else:
        fc1 = Dense(classes, name='fc' + str(classes), activation='softmax')(conv_in)

    return Model(d0, fc1)


def weedann(img_shape=(256, 256, 50, 1), classes=2):
    # input layer
    input_layer = Input(img_shape)

    fc = Dense(units=500, activation='relu', kernel_regularizer=regularizers.l2(0.005))(input_layer) # , kernel_regularizer=regularizers.l2(0.005)
    # fc = Dropout(0.2)(fc)
    fc = Dense(units=500, activation='relu', kernel_regularizer=regularizers.l2(0.005))(fc)
    # fc = Dropout(0.5)(fc)
    # fc = Dense(units=500, activation='relu')(fc)
    fc = Dense(units=classes, activation='softmax')(fc)

    return Model(input_layer, fc)

            
l = hyper3dnet((25, 25, 300, 1), classes=3)