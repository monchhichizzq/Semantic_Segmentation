# -*- coding: utf-8 -*-
# @Time    : 2021/2/6 22:12
# @Author  : Zeqi@@
# @FileName: transpose_conv.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2DTranspose

# Stride 1
input = [[[1, 3, 2, 1], [1, 3, 3, 1], [2, 1, 1, 3], [3, 2, 3, 3]]]
w = [[1, 2, 3], [0, 1, 0], [2, 1, 2]]
input = np.expand_dims(input, axis=-1)
input = tf.constant(input, dtype=tf.float32)
print('input shape: ', np.shape(input))
w = tf.keras.initializers.constant(w)
print('weight shape: ', np.shape(w))
Tconv = Conv2DTranspose(filters = 1,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=w,
                        use_bias=False)

out = Tconv(input)
print('Transpose Conv Out: ', out)


# Stride 2
input = [[[3, 3], [1, 1]]]
w = [[1, 2, 3], [0, 1, 0], [2, 1, 2]]
input = np.expand_dims(input, axis=-1)
input = tf.constant(input, dtype=tf.float32)
print('input shape: ', np.shape(input))
w = tf.keras.initializers.constant(w)
print('weight shape: ', np.shape(w))
Tconv = Conv2DTranspose(filters = 1,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=w,
                        use_bias=False)

out = Tconv(input)
print('Transpose Conv Out: ', out)


