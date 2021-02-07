# -*- coding: utf-8 -*-
# @Time    : 2021/2/7 1:38
# @Author  : Zeqi@@
# @FileName: MobileNet.py
# @Software: PyCharm

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
from tensorflow.keras.models import Model


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Backbone - MobileNetV1 -Unet')

class MobileNetV1:
    def __init__(self, add_bias, add_bn):
        self.add_bias = add_bias
        self.add_bn = add_bn


    def _conv_block(self, inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
        filters = int(filters * alpha)
        x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
        x = Conv2D(filters, kernel, padding='valid',
                                    use_bias=self.add_bias,
                                    strides=strides,
                                    name='conv1')(x)
        if self.add_bn:
            x = BatchNormalization(name='conv1_bn')(x)

        x = ReLU(max_value=6, name='conv1_relu')(x)
        return x

    def _depthwise_conv_block(self, inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)

        x = ZeroPadding2D((1, 1), name='conv_pad_%d' % block_id)(inputs)
        x = DepthwiseConv2D((3, 3), padding='valid', # Unet 特有 valid
                            depth_multiplier=depth_multiplier,
                            strides=strides,
                            use_bias=self.add_bias,
                            name='conv_dw_%d' % block_id)(x)

        if self.add_bn:
            x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
        x = ReLU(max_value=6, name='conv_dw_%d_relu' % block_id)(x)

        x = Conv2D(pointwise_conv_filters, (1, 1),
                            padding='same',
                            use_bias=self.add_bias,
                            strides=(1, 1),
                            name='conv_pw_%d' % block_id)(x)

        if self.add_bn:
            x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)

        x = ReLU(max_value=6, name='conv_pw_%d_relu' % block_id)(x)
        return x

    def __call__(self, img_input, pretrained=True, *args, **kwargs):
        alpha=1.0
        depth_multiplier=1

        # 416,416,3 -> 208,208,32 -> 208,208,64
        x = self._conv_block(img_input, 32, alpha, strides=(2, 2))
        x = self._depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
        f1 = x

        # 208,208,64 -> 104,104,128
        x = self._depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
        x = self._depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
        f2 = x

        # 104,104,128 -> 52,52,256
        x = self._depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
        x = self._depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
        f3 = x

        # 52,52,256 -> 26,26,512
        x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
        x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
        x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
        x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
        x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
        x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
        f4 = x

        # 26,26,512 -> 13,13,1024
        x = self._depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
        x = self._depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
        f5 = x

        if pretrained:
            mbv1_model = tf.keras.applications.MobileNet(include_top=False, weights='imagenet')
            mbv1 = mbv1_model.get_weights()

            mbv1_backbone = Model(img_input, x)
            model_w = mbv1_backbone.get_weights()

            for i, w in enumerate(model_w):
                model_w[i] = mbv1[i]
                logger.info('{} backbone: {} load the weights of pretrained vgg16_224: {}'.format(i, np.shape(w), np.shape(mbv1[i])))
                if i >= len(mbv1) - 1:
                    break
            mbv1_backbone.set_weights(model_w)

        return img_input, [f1 , f2 , f3 , f4 , f5]