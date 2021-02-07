# -*- coding: utf-8 -*-
# @Time    : 2021/2/7 1:32
# @Author  : Zeqi@@
# @FileName: unet.py
# @Software: PyCharm

import sys
sys.path.append('../../../../Semantic_Segmentation')

import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import  DepthwiseConv2D
from tensorflow.keras.layers import  UpSampling2D
from tensorflow.keras.layers import  Concatenate
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Layer

from tensorflow.keras.models import Model

from models.tensorflow.Unet.MobileNet import MobileNetV1
from models.tensorflow.Unet.VGG16 import get_vgg_encoder

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Decoder - Unet')


class CropConcatBlock(Layer):

    def call(self, inputs , **kwargs):
        down_layer, x = inputs
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:,
                             height_diff: (x2_shape[1] + height_diff),
                             width_diff: (x2_shape[2] + width_diff),
                             :]

        x = Concatenate(axis=-1)([down_layer_cropped, x])
        return x

class Unet_model:
    def __init__(self, add_bias, add_bn, num_classes=21, padding ='valid', backbone='MobileNet'):
        self.add_bias = add_bias
        self.add_bn = add_bn
        self.num_classes = num_classes
        self.backbone = backbone
        self.padding = padding
        self.channels = [64, 128, 256, 512]

    def up_block(self, inputs, pool_size=2, up_operation = 'UpSampling2D'):
        '''
            Up-Sampling block in Unet decoder
        :param inputs: Tensor, Feature maps
        :param pool_size: int, kernel_size, stride
        :param up_operation: str, 'UpSampling2D' or 'Transpose_conv2d'
        :return:
        '''
        if up_operation is 'UpSampling2D':
            x = UpSampling2D(size=(pool_size, pool_size))(inputs)
        elif up_operation is 'Transpose_conv2d':
            filters = inputs.shape[-1]
            x = Conv2DTranspose(filters // 2,
                                 kernel_size=(pool_size, pool_size),
                                 kernel_initializer='he_normal',
                                 strides=pool_size,
                                 padding="valid")(inputs)
        return x


    def __call__(self, inputs,  up_operation = 'UpSampling2D', *args, **kwargs):
        # -------------------------------#
        #   feat1   512,512,64
        #   feat2   256,256,128
        #   feat3   128,128,256
        #   feat4   64,64,512
        #   feat5   32,32,512
        # -------------------------------#


        if self.backbone is 'MobileNet':
            backbone = MobileNetV1(add_bias= self.add_bias, add_bn=self.add_bn)
            img_input, [feat1, feat2, feat3, feat4, feat5] = backbone(inputs)

        if self.backbone is 'VGG16':
            img_input, [feat1, feat2, feat3, feat4, feat5] = get_vgg_encoder(inputs, self.add_bias, padding='valid')

        # 32, 32, 512 -> 64, 64, 512
        # P5_up = UpSampling2D(size=(2, 2))(feat5)
        P5_up = self.up_block(feat5, pool_size=2, up_operation = up_operation)

        # 64, 64, 512 + 64, 64, 512 -> 64, 64, 1024
        # P4 = Concatenate(axis=3)([feat4, P5_up])
        P4 = CropConcatBlock()([feat4, P5_up])
        # 64, 64, 1024 -> 64, 64, 512
        P4 = Conv2D(self.channels[3], 3, activation='relu', padding=self.padding, kernel_initializer='he_normal')(P4)
        P4 = Conv2D(self.channels[3], 3, activation='relu', padding=self.padding, kernel_initializer='he_normal')(P4)

        # 64, 64, 512 -> 128, 128, 512
        # P4_up = UpSampling2D(size=(2, 2))(P4)
        P4_up = self.up_block(P4, pool_size=2, up_operation=up_operation)
        # 128, 128, 256 + 128, 128, 512 -> 128, 128, 768
        # P3 = Concatenate(axis=3)([feat3, P4_up])
        P3 = CropConcatBlock()([feat3, P4_up])
        # 128, 128, 768 -> 128, 128, 256
        P3 = Conv2D(self.channels[2], 3, activation='relu', padding=self.padding, kernel_initializer='he_normal')(P3)
        P3 = Conv2D(self.channels[2], 3, activation='relu', padding=self.padding, kernel_initializer='he_normal')(P3)

        # 128, 128, 256 -> 256, 256, 256
        # P3_up = UpSampling2D(size=(2, 2))(P3)
        P3_up = self.up_block(P3, pool_size=2, up_operation=up_operation)
        # 256, 256, 256 + 256, 256, 128 -> 256, 256, 384
        # P2 = Concatenate(axis=3)([feat2, P3_up])
        P2 = CropConcatBlock()([feat2, P3_up])
        # 256, 256, 384 -> 256, 256, 128
        P2 = Conv2D(self.channels[1], 3, activation='relu', padding=self.padding, kernel_initializer='he_normal')(P2)
        P2 = Conv2D(self.channels[1], 3, activation='relu', padding=self.padding, kernel_initializer='he_normal')(P2)

        # 256, 256, 128 -> 512, 512, 128
        #  P2_up = UpSampling2D(size=(2, 2))(P2)
        P2_up = self.up_block(P2, pool_size=2, up_operation=up_operation)
        # 512, 512, 128 + 512, 512, 64 -> 512, 512, 192
        P1 = CropConcatBlock()([feat1, P2_up])
        # P1 = Concatenate(axis=3)([feat1, P2_up])
        # 512, 512, 192 -> 512, 512, 64
        P1 = Conv2D(self.channels[0], 3, activation='relu', padding=self.padding, kernel_initializer='he_normal')(P1)
        P1 = Conv2D(self.channels[0], 3, activation='relu', padding=self.padding, kernel_initializer='he_normal')(P1)

        # 512, 512, 64 -> 512, 512, num_classes
        P1 = Conv2D(self.num_classes, 1, activation="softmax")(P1)

        model = Model(inputs=inputs, outputs=P1)
        return model

if __name__ == '__main__':
    input_shape = (572, 572, 3)
    MobileNet_Unet = Unet_model(add_bias=False, add_bn=True, num_classes=2, backbone='MobileNet')
    inputs = Input(input_shape)
    model = MobileNet_Unet(inputs, up_operation = 'UpSampling2D')
    model.summary()

    input_shape = (572, 572, 3)
    VGG_Unet = Unet_model(add_bias=True, add_bn=False, num_classes=2, backbone='VGG16')
    inputs = Input(input_shape)
    model = VGG_Unet(inputs, up_operation = 'Transpose_conv2d')
    model.summary()
    model.save('vgg_unet.h5')


'''
UpSampling2D is just a simple scaling up of the image by using nearest neighbour or bilinear upsampling, so nothing smart. Advantage is it's cheap.

Conv2DTranspose is a convolution operation whose kernel is learnt (just like normal conv2d operation) while training your model. 
Using Conv2DTranspose will also upsample its input but the key difference is the model should learn what is the best upsampling for the job.
'''


'''
Model size
output class
copy and crop
Unet++
padding valid
replace upconv
maxpooling

'''