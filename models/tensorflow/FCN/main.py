# -*- coding: utf-8 -*-
# @Time    : 2021/2/6 18:39
# @Author  : Zeqi@@
# @FileName: main.py.py
# @Software: PyCharm

import sys
sys.path.append('../FCN')

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Softmax
from tensorflow.keras.models import Model
from backbone import get_vgg_encoder


class FCN8s():
    def __init__(self, backbone, add_bias = True, add_bn = False, n_classes=20):
        self.backbone     = backbone
        self.add_bn        = add_bn
        self.add_bias     = add_bias

        self.filters_dir  = {'block6_fc1_bn'  : 4096,
                             'block6_fc2_bn'  : 4096,
                             'block7_fc1_bn'  : n_classes,
                             'Upsample'     : n_classes,
                             'ScaleDown16_b'  : n_classes,
                             'ScaleDown8_b': n_classes,
                             'ScaleDown1_b': n_classes}

        self.initializer = tf.keras.initializers.he_normal()

    def conv2d_bn(self,
                  conv_ind,
                  x,
                  filters,
                  num_row,
                  num_col,
                  padding='same',
                  stride=1,
                  dilation_rate=1,
                  relu=True,
                  relu6=True,
                  max_pool=False,
                  l1=None):

        if l1 is not None:
            l1_regularizer = tf.keras.regularizers.l1(l1)
            # print('L1 regularization - l1 coefficient: {}'.format(l1_coeff))
        else:
            l1_regularizer = None
            # print('Not use regularization')

        x = Conv2D(filters            = filters,
                   kernel_size        = (num_row, num_col),
                   strides            = (stride, stride),
                   padding            = padding,
                   dilation_rate      = (dilation_rate, dilation_rate),
                   use_bias           = self.add_bias,
                   kernel_initializer = self.initializer,
                   kernel_regularizer = l1_regularizer,
                   name               = 'block{}_conv2d_{}'.format(conv_ind[0], conv_ind[1]))(x)

        if self.add_bn:
            x = BatchNormalization(name='block{}_conv2d_{}_bn'.format(conv_ind[0], conv_ind[1]))(x)

        if relu:
            if relu6:
                x = ReLU(max_value=6, name='block{}_conv2d_{}_relu6'.format(conv_ind[0], conv_ind[1]))(x)
            else:
                x = ReLU(name='block{}_conv2d_{}_relu'.format(conv_ind[0], conv_ind[1]))(x)

        if max_pool:
           x =MaxPooling2D((2, 2), strides=(2, 2), name='block{}_conv2d_{}_bn_pool'.format(conv_ind[0], conv_ind[1]))(x)

        return x


    def __call__(self, input_tensor, *args, **kwargs):
        pretrained = kwargs.get('pretrained', True)

        if self.backbone == 'vgg16':
            img_input, [f1, f2, f3, f4, f5] = get_vgg_encoder(input_tensor, self.add_bias, pretrained=pretrained)

        # Dense layer ->  Convolutinalized fully connected layer.
        # 4, 8, 512 -> 4, 8, 4096
        x = self.conv2d_bn(conv_ind             = [6, 1],
                           x                    = f5,
                           filters              = self.filters_dir['block6_fc1_bn'],
                           num_row              = 7,
                           num_col              = 7,
                           relu                 = True,
                           relu6                = False,
                           max_pool             = False,
                           l1                   = None)

        # 4, 8, 4096 -> 4, 8, 4096
        x = self.conv2d_bn(conv_ind             = [6, 2],
                           x                    = x,
                           filters              = self.filters_dir['block6_fc2_bn'],
                           num_row              = 1,
                           num_col              = 1,
                           relu                 = True,
                           relu6                = False,
                           max_pool             = False,
                           l1                   = None)

        # Classifying layers.
        # 4, 8, 4096 -> 4, 8, 20
        out_ScaleDown_32 = self.conv2d_bn(conv_ind             = [7, 1],
                                           x                    = x,
                                           filters              = self.filters_dir['block7_fc1_bn'],
                                           num_row              = 1,
                                           num_col              = 1,
                                           relu                 = False,
                                           relu6                = False,
                                           max_pool             = False,
                                           l1                   = None)

        # Upsampling 4, 8, 20 -> 8, 16, 20
        out_ScaleDown_16_a = Conv2DTranspose(self.filters_dir['Upsample'],
                                            kernel_size=(4, 4),
                                            strides=(2, 2),
                                            padding='same',
                                            use_bias=False)(out_ScaleDown_32)

        # 8, 16, 512 -> 8, 16, 20
        out_ScaleDown_16_b = self.conv2d_bn(conv_ind             = [8, 1],
                                           x                    = f4,
                                           filters              = self.filters_dir['ScaleDown16_b'],
                                           num_row              = 1,
                                           num_col              = 1,
                                           relu                 = True,
                                           relu6                = False,
                                           max_pool             = False,
                                           l1                   = None)
        # 8, 16, 20
        out_ScaleDown_16 = Add()([out_ScaleDown_16_a, out_ScaleDown_16_b])

        # Upsampling 8, 16, 20 -> 16, 32, 20
        out_ScaleDown_8_a = Conv2DTranspose(self.filters_dir['Upsample'],
                                            kernel_size=(4, 4),
                                            strides=(2, 2),
                                            padding='same',
                                            use_bias=False)(out_ScaleDown_16)
        # 16, 32, 256 -> 16, 32, 20
        out_ScaleDown_8_b = self.conv2d_bn(conv_ind             = [8, 2],
                                           x                   = f3,
                                           filters             = self.filters_dir['ScaleDown8_b'],
                                           num_row             = 1,
                                           num_col             = 1,
                                           relu                = True,
                                           relu6               = False,
                                           max_pool            = False,
                                           l1                  = None)
        # 16, 32, 20
        out_ScaleDown_8 = Add()([out_ScaleDown_8_a,out_ScaleDown_8_b])


        # Upsampling 16, 32, 20 -> 128, 256, 20
        out_ScaleDown_1 = Conv2DTranspose(self.filters_dir['Upsample'],
                                               kernel_size=(16, 16),
                                               strides=(8, 8),
                                               padding='same',
                                               use_bias=False)(out_ScaleDown_8)

        out = Softmax(axis=-1)(out_ScaleDown_1)

        model = Model(input_tensor, out, name=self.backbone + '-FCNs8')
        return model


if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    input_tensor = Input(shape=(224, 224, 3))
    FCN = FCN8s(backbone='vgg16', add_bias=True, add_bn=False)
    model = FCN(input_tensor)
    model.summary()




