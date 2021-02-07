# -*- coding: utf-8 -*-
# @Time    : 2021/2/7 1:37
# @Author  : Zeqi@@
# @FileName: VGG16.py
# @Software: PyCharm


import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Backbone - VGG16 -Unet')


def get_vgg_encoder(img_input, use_bias, padding, pretrained=False):

    # assert input_height % 32 == 0
    # assert input_width % 32 == 0

    # Block 1
    x = Conv2D(64, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block3_conv2')(x)
    # x = Conv2D(256, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block4_conv2')(x)
    # x = Conv2D(512, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block5_conv1')(x)
    #x = Conv2D(512, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block5_conv2')(x)
    #x = Conv2D(512, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Conv2D(1024, (3, 3), use_bias=use_bias, activation='relu', padding=padding, name='block5_conv2')(x)
    f5 = x

    if pretrained:
        vgg16_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        vgg16_224 = vgg16_model.get_weights()

        fcn_vgg16 = Model(img_input, x)
        model_w = fcn_vgg16.get_weights()

        for i, w in enumerate(model_w):
            model_w[i] = vgg16_224[i]
            logger.info('{} backbone: {} load the weights of pretrained vgg16_224: {}'.format(i, np.shape(w), np.shape(vgg16_224[i])))
            if i >= len(vgg16_224) - 1:
                break
        fcn_vgg16.set_weights(model_w)

    return img_input, [f1, f2, f3, f4, f5]


if __name__ == '__main__':
    img_input = Input(shape=(572, 572, 3))
    img_input, [f1, f2, f3, f4, f5] = get_vgg_encoder(img_input, use_bias=True, pretrained=False, padding='valid')
    vgg16 = Model(img_input, f5)
    vgg16.summary()