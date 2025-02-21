"""
Apache License Version 2.0
https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/diffjpeg.py
"""

import itertools
from typing import Optional, Union

import numpy as np
import tensorflow as tf

# todo : BaseImagePreprocessingLayer is inherited tf.Module
# from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import BaseImagePreprocessingLayer

# ------------------------ utils ------------------------#
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

y_table = tf.constant(y_table, dtype=tf.float32)
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = tf.constant(c_table, dtype=tf.float32)


def diff_round(x):
    """ Differentiable rounding function
    """
    return tf.round(x) + (x - tf.round(x)) ** 3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality

    Args:
        quality(float): Quality for jpeg compression.

    Returns:
        float: Compression factor.
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.


# ------------------------ compression ------------------------#
class RGB2YCbCrJpeg(tf.Module):
    """ Converts RGB image to YCbCr
    """

    def __init__(self, **kwargs):
        super(RGB2YCbCrJpeg, self).__init__(**kwargs)
        matrix = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
                          dtype=np.float32).T
        self.shift = tf.constant([0., 128., 128.], dtype=tf.float32)
        self.matrix = tf.constant(matrix, dtype=tf.float32)

    def __call__(self, image):
        """
        Args:
            image(Tensor): [b,h,w,c]
        Returns:
            Tensor: batch x height x width x 3
        """
        result = tf.tensordot(image, self.matrix, axes=1) + self.shift
        return result


class ChromaSubsampling(tf.Module):
    """ Chroma subsampling on CbCr channels
    """

    def __init__(self, **kwargs):
        super(ChromaSubsampling, self).__init__(**kwargs)

    def __call__(self, image):
        """
        Args:
            image(Tensor): [b,h,w,c]

        Returns:
            y(tensor): batch x height x width
            cb(tensor): batch x height/2 x width/2
            cr(tensor): batch x height/2 x width/2
        """
        # image_2 = image.permute(0, 3, 1, 2).clone()

        cb = tf.nn.avg_pool2d(image[:, :, :, 1:2], ksize=2, strides=[1, 2, 2, 1], padding='SAME')
        cr = tf.nn.avg_pool2d(image[:, :, :, 2:3], ksize=2, strides=[1, 2, 2, 1], padding='SAME')

        return image[:, :, :, 0], cb[:, :, :, 0], cr[:, :, :, 0]


class BlockSplitting(tf.Module):
    """ Splitting image into patches
    """

    def __init__(self, k=8, **kwargs):
        super(BlockSplitting, self).__init__(**kwargs)
        self.k = k

    def __call__(self, image):
        """
        Args:
            image(tensor): [b,h,w]

        Returns:
            Tensor:  batch x h*w/64 x h x w
        """
        input_shape = tf.shape(image)
        b, h, w = input_shape[0], input_shape[1], input_shape[2]
        image_reshaped = tf.reshape(image, [b, h // self.k, self.k, -1,
                                            self.k])  # image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = tf.transpose(image_reshaped, perm=[0, 1, 3, 2, 4])
        return tf.reshape(image_transposed, [b, -1, self.k, self.k])


class DCT8x8(tf.Module):
    """ Discrete Cosine Transformation
    """

    def __init__(self, **kwargs):
        super(DCT8x8, self).__init__(**kwargs)
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.tensor = tf.constant(tensor, dtype=tf.float32)
        self.scale = tf.constant(np.outer(alpha, alpha) * 0.25, dtype=tf.float32)

    def __call__(self, image):
        """
        Args:
            image(tensor): [b, h*w/64, h, w]

        Returns:
            Tensor: batch x height x width
        """
        image = image - 128
        result = self.scale * tf.tensordot(image, self.tensor, axes=2)
        # result = keras.ops.reshape(result, image.shape)
        return result


class YQuantize(tf.Module):
    """ JPEG Quantization for Y channel

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding, **kwargs):
        super(YQuantize, self).__init__(**kwargs)
        self.rounding = rounding
        self.y_table = y_table

    def __call__(self, image, factor=1.0):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            # todo: tf.float32 vs tf.float64
            image = tf.cast(image, dtype=tf.float32) / (self.y_table * factor)
        else:
            table = self.y_table[None, None] * factor[:, None, None, None]
            image = tf.cast(image, dtype=tf.float32) / table
        image = self.rounding(image)
        return image


class CQuantize(tf.Module):
    """ JPEG Quantization for CbCr channels

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding, **kwargs):
        super(CQuantize, self).__init__(**kwargs)
        self.rounding = rounding
        self.c_table = c_table

    def __call__(self, image, factor=1.0):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            image = tf.cast(image, dtype=tf.float32) / (self.c_table * factor)
        else:
            table = self.c_table[None, None] * factor[:, None, None, None]
            image = tf.cast(image, dtype=tf.float32) / table
        image = self.rounding(image)
        return image


class CompressJpeg(tf.Module):
    """Full JPEG compression algorithm

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding=tf.round, **kwargs):
        super(CompressJpeg, self).__init__(**kwargs)

        self.l1 = RGB2YCbCrJpeg()
        self.l2 = ChromaSubsampling()
        self.l3 = BlockSplitting()
        self.l4 = DCT8x8()
        self.c_quantize = CQuantize(rounding=rounding)
        self.y_quantize = YQuantize(rounding=rounding)

    def __call__(self, image, factor=1.0):
        """
        Args:
            image(tensor): batch x 3 x height x width

        Returns:
            dict(tensor): Compressed tensor with batch x h*w/64 x 8 x 8.
        """
        image = self.l1(image * 255)
        y, cb, cr = self.l2(image)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l3(components[k])
            comp = self.l4(comp)
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp, factor=factor)
            else:
                comp = self.y_quantize(comp, factor=factor)

            components[k] = comp

        return components['y'], components['cb'], components['cr']


# ------------------------ decompression ------------------------#


class YDequantize(tf.Module):
    """Dequantize Y channel
    """

    def __init__(self, **kwargs):
        super(YDequantize, self).__init__(**kwargs)
        self.y_table = y_table

    def __call__(self, image, factor=1.0):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            out = image * (self.y_table * factor)
        else:
            table = self.y_table[None, None] * factor[:, None, None, None]
            out = image * table
        return out


class CDequantize(tf.Module):
    """Dequantize CbCr channel
    """

    def __init__(self, **kwargs):
        super(CDequantize, self).__init__(**kwargs)
        self.c_table = c_table

    def __call__(self, image, factor=1.0):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            out = image * (self.c_table * factor)
        else:
            table = self.c_table[None, None] * factor[:, None, None, None]
            out = image * table
        return out


class iDCT8x8(tf.Module):
    """Inverse discrete Cosine Transformation
    """

    def __init__(self, **kwargs):
        super(iDCT8x8, self).__init__(**kwargs)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = tf.constant(np.outer(alpha, alpha), dtype=tf.float32)
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos((2 * v + 1) * y * np.pi / 16)
        self.tensor = tf.constant(tensor, dtype=tf.float32)

    def __call__(self, image):
        """
        Args:
            image(tensor): [b, h, w]

        Returns:
            Tensor: batch x height x width
        """
        image = image * self.alpha
        result = 0.25 * tf.tensordot(image, self.tensor, axes=2) + 128
        # result = keras.ops.reshape(result, image.shape)
        return result


class BlockMerging(tf.Module):
    """Merge patches into image
    """

    def __init__(self, **kwargs):
        super(BlockMerging, self).__init__(**kwargs)

    def __call__(self, patches, height=None, width=None):
        """
        Args:
            patches(tensor) [b, hxw/64, hxw]
            height(int)
            width(int)

        Returns:
            Tensor: batch x height x width
        """
        k = 8
        input_shape = tf.shape(patches)
        b = input_shape[0]

        image_reshaped = tf.reshape(patches, [b, height // k, width // k, k, k])
        image_transposed = tf.transpose(image_reshaped, perm=[0, 1, 3, 2, 4])
        return tf.reshape(image_transposed, [b, height, width])


class ChromaUpsampling(tf.Module):
    """Upsample chroma layers
    """

    def __init__(self, **kwargs):
        super(ChromaUpsampling, self).__init__(**kwargs)

    def __call__(self, inputs):
        """
        Args:
            inputs :
                y(tensor): [b,h,w] y channel image
                cb(tensor): [b,h,w] cb channel
                cr(tensor): [b,h,w] cr channel

        Returns:
            Tensor: batch x height x width x 3
        """
        y, cb, cr = inputs

        def repeat(x, k=2):
            input_shape= tf.shape(x)
            height, width = input_shape[1], input_shape[2]
            x = x[:, :, :, None]
            # x = keras.ops.repeat(x, repeats=k, axis=3)
            # todo : repeat torch와 다른것 해결 .
            x = tf.tile(x, multiples=[1, 1, k, k])
            x = tf.reshape(x, [-1, height * k, width * k])
            return x

        cb = repeat(cb)
        cr = repeat(cr)
        return tf.concat([y[:, :, :, None], cb[:, :, :, None], cr[:, :, :, None]], axis=3)


class YCbCr2RGBJpeg(tf.Module):
    """Converts YCbCr image to RGB JPEG
    """

    def __init__(self, **kwargs):
        super(YCbCr2RGBJpeg, self).__init__(**kwargs)

        matrix = np.array([[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]], dtype=np.float32).T
        self.shift = tf.constant([0, -128., -128.], dtype=tf.float32)
        self.matrix = tf.constant(matrix, dtype=tf.float32)

    def __call__(self, image):
        """
        Args:
            image(tensor): [b, h, w, c]

        Returns:
            Tensor: [b, h, w, c]
        """
        result = tf.tensordot(image + self.shift, self.matrix, axes=1)
        # result = keras.ops.transpose(result, axes=[0,3,1,2])
        # todo : 같은 shape 으로 reshape하는게 무슨 의미지 ..?
        # result = keras.ops.reshape(result, image.shape)
        # keras.ops.transpose
        return result


class DeCompressJpeg(tf.Module):
    """Full JPEG decompression algorithm

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding=tf.round, **kwargs):
        super(DeCompressJpeg, self).__init__(**kwargs)
        self.c_dequantize = CDequantize()
        self.y_dequantize = YDequantize()
        self.idct = iDCT8x8()
        self.merging = BlockMerging()
        self.chroma = ChromaUpsampling()
        self.colors = YCbCr2RGBJpeg()

    def __call__(self, inputs, imgh=None, imgw=None, factor=1.0):
        """
        Args:
            inputs :
                y,
                cb,
                cr,
            imgh(int)
            imgw(int)
            factor(float)

        Returns:
            Tensor: batch x 3 x height x width
        """
        y, cb, cr = inputs
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k], factor=factor)
                height, width = int(imgh / 2), int(imgw / 2)
            else:
                comp = self.y_dequantize(components[k], factor=factor)
                height, width = imgh, imgw
            comp = self.idct(comp)
            components[k] = self.merging(comp, height=height, width=width)
            #
        image = self.chroma([components['y'], components['cb'], components['cr']])
        image = self.colors(image)
        r_term = tf.where(tf.zeros_like(image) < image, image, 0)
        l_term = 255 * tf.ones_like(image)

        # get minimum value.
        image = tf.where(l_term < r_term, l_term, r_term)
        return image / 255


# ------------------------ main DiffJPEG ------------------------ #

class DiffJPEG(tf.Module):
    """This JPEG algorithm result is slightly different from cv2.
    DiffJPEG supports batch processing.

    Args:
        shape(list) : Shape of image.
        differentiable(bool): If True, uses custom differentiable rounding function, if False, uses standard torch.round
    """

    def __init__(self, differentiable=True, **kwargs):
        super(DiffJPEG, self).__init__(**kwargs)
        if differentiable:
            rounding = diff_round
        else:
            rounding = tf.round

        self.compress = CompressJpeg(rounding=rounding)
        self.decompress = DeCompressJpeg(rounding=rounding)

    def __call__(self, x, quality: Union[float, tf.Tensor] = 1.0):
        """
        Args:
            x (Tensor): Input image, bchw, rgb, [0, 1]
            quality(float): Quality factor for jpeg compression scheme.
        """
        factor = quality
        if isinstance(factor, (int, float)):
            factor = quality_to_factor(factor)
        else:
            for i in range(factor.shape[0]):
                factor[i] = quality_to_factor(factor[i])
        input_shape = tf.shape(x)
        h, w = input_shape[1], input_shape[2]
        h_pad, w_pad = 0, 0
        # why should use 16
        if h % 16 != 0:
            h_pad = 16 - h % 16
        if w % 16 != 0:
            w_pad = 16 - w % 16
        x = tf.image.pad_to_bounding_box(x,
                                         offset_width=0,
                                         target_width=w + w_pad,
                                         offset_height=0,
                                         target_height=h + h_pad)  # zero-padding

        y, cb, cr = self.compress(x, factor=factor)
        recovered = self.decompress([y, cb, cr], imgh=(h + h_pad), imgw=(w + w_pad), factor=factor)
        recovered = recovered[:, 0:h:, 0:w, :]
        return recovered
