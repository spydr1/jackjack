"""
Apache License Version 2.0
https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/img_process_util.py
"""

import cv2
import numpy as np
import tensorflow as tf
import tf_keras as keras
from sympy import transpose


# This code differ to origin code.
# All kernel is not "batched", it means that all image processed by single kernel.

# 모든 배치에서 random kernel 이 생성되는데 이를 굳이 배치안의 모든 이미지를 다르게 처리할 필요가 없다고 생각해서.


def filter2D(img: tf.Tensor, kernel=None):
    """Tensorflow version of cv2.filter2D
    Args:
        img (Tensor): (b, h, w, c)
        kernel (Tensor): (k, k)
    """
    kernel = tf.cast(kernel, img.dtype)
    k = kernel.shape[-1]
    b, h, w, c = img.shape
    if k % 2 == 1:
        img = tf.pad(img, [[0, 0],
                           [k // 2, k // 2],
                           [k // 2, k // 2],
                           [0, 0]],
                     mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    kernel = tf.repeat(kernel[:, :, None, None], repeats=c, axis=2)
    filtered_img = tf.nn.depthwise_conv2d(img, kernel, strides=[1,1,1,1], padding="VALID")

    return filtered_img

def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.

    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I


    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img


class USMSharp(keras.layers.Layer):

    def __init__(self, radius=50, sigma=0, **kwargs):
        super(USMSharp, self).__init__(**kwargs)
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        self.kernel = tf.constant(np.dot(kernel, kernel.transpose()), dtype=tf.float32)

    def call(self, img, weight=0.5, threshold=10):
        # b, h, w, c = img.shape
        # new_kernel = keras.ops.repeat(self.kernel, b, axis=0)
        blur = filter2D(img, kernel=self.kernel)
        residual = img - blur

        mask = tf.abs(residual) * 255 > threshold
        mask = tf.cast(mask, img.dtype)
        soft_mask = filter2D(mask, kernel=self.kernel)
        sharp = img + weight * residual
        sharp = tf.clip_by_value(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img
