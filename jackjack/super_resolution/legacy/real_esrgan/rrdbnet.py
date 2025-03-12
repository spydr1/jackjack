"""
BSD 3-Clause License
https://github.com/ai-forever/Real-ESRGAN/blob/main/RealESRGAN/model.py
"""
import os
import logging
import pathlib
import tqdm

import kagglehub
import tensorflow as tf
import tensorflow.keras as keras

from jackjack.super_resolution.config import keras_home

initializers = keras.initializers
Layer = keras.layers.Layer
Conv2D = keras.layers.Conv2D
LeakyReLU = keras.layers.LeakyReLU
UpSampling2D = keras.layers.UpSampling2D
Input = keras.layers.Input

from jackjack.super_resolution.legacy.real_esrgan.image_utils import *

class ResidualDenseBlock(Layer):
    def __init__(self, output_dim=64, hidden_dim=32, **kwargs):
        super().__init__(**kwargs)
        kernel_init = initializers.HeNormal()
        bias_init = initializers.Constant(value=0.1)
        conv_num = 4
        self.conv_list = [Conv2D(filters=hidden_dim,
                                 kernel_size=3,
                                 padding="same",
                                 kernel_initializer=kernel_init,
                                 bias_initializer=bias_init,
                                 name=f'conv{i}') for i in range(1, conv_num + 1)
                          ]
        self.last_conv = Conv2D(filters=output_dim,
                                kernel_size=3,
                                padding="same",
                                kernel_initializer=kernel_init,
                                bias_initializer=bias_init,
                                name='conv5'
                                )
        if tf._tf_uses_legacy_keras :
            self.lrelu = LeakyReLU(alpha=0.2)
        else :
            self.lrelu = LeakyReLU(negative_slope=0.2)

        # initialization
        # https://github.com/ai-forever/Real-ESRGAN/blob/main/RealESRGAN/arch_utils.py#L9
        # default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        # this code means that kaiming normal & scale=1.0 bias filled by 0.1

    def call(self, x):
        x1 = self.lrelu(self.conv_list[0](x))
        x2 = self.lrelu(self.conv_list[1](tf.concat((x, x1), -1)))
        x3 = self.lrelu(self.conv_list[2](tf.concat((x, x1, x2), -1)))
        x4 = self.lrelu(self.conv_list[3](tf.concat((x, x1, x2, x3), -1)))
        x5 = self.last_conv(tf.concat((x, x1, x2, x3, x4), -1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(Layer):
    """Residual in Residual Dense Block.
    Used in RRDB-Net in ESRGAN.
    Args:
        hidden_dim (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, output_dim=64, hidden_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.rdb1 = ResidualDenseBlock(output_dim, hidden_dim, name='rdb1')
        self.rdb2 = ResidualDenseBlock(output_dim, hidden_dim, name='rdb2')
        self.rdb3 = ResidualDenseBlock(output_dim, hidden_dim, name='rdb3')

    def call(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

def is_exp_binary(n):
    return n & (n - 1) == 0


class RRDBNet(keras.Model):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        img_height (int) :
        img_width (int) :
        scale (int) :
        hidden_dim (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        body_hidden_dim (int): Channels for each growth. Default: 32.
    """
    def __init__(self, img_height=240, img_width=240, upscale=4, hidden_dim=64, num_block=23, body_hidden_dim=32,):

        # assert is_exp_binary(upscale), "upscale must be power of 2."
        assert upscale in [2, 4, 8], "upscale must be 2,4,8."
        assert img_height % upscale == 0 and img_width % upscale == 0, "image resolution must be multiply of 2."
        self.upscale = upscale

        unshuffle_scale = 1
        if upscale == 2:
            unshuffle_scale = 2

        image_channel = 3 * unshuffle_scale ** 2
        img_height = img_height // unshuffle_scale
        img_width = img_width // unshuffle_scale

        low_resolution_image = Input((img_height, img_width, image_channel), name="input_image")
        x = low_resolution_image
        x = residual = Conv2D(filters=hidden_dim, kernel_size=3, strides=1, padding="same", name='conv_first')(x)
        for i in range(num_block):
            x = RRDB(output_dim=hidden_dim, hidden_dim=body_hidden_dim, name=f'body.{i}')(x)
        x = Conv2D(filters=hidden_dim, kernel_size=3, strides=1, padding="same", name='conv_body')(x)
        x = x + residual

        upscale_num = 4 if upscale == 8 else 3
        for i in range(1, upscale_num):
            x = UpSampling2D(size=(2, 2), interpolation="nearest")(x)
            x = Conv2D(filters=hidden_dim, kernel_size=3, strides=1, padding="same", name=f'conv_up{i}')(x)
            if tf._tf_uses_legacy_keras:
                x = LeakyReLU(alpha=0.2)(x)
            else:
                x = LeakyReLU(negative_slope=0.2)(x)
        # if x8

        x = Conv2D(filters=hidden_dim, kernel_size=3, strides=1, padding="same", name='conv_hr')(x)
        if tf._tf_uses_legacy_keras:
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = LeakyReLU(negative_slope=0.2)(x)
        output = Conv2D(filters=3, kernel_size=3, strides=1, padding="same", name='conv_last')(x)

        super().__init__(low_resolution_image, output)


    def to_quant(self):
        tf_callable = tf.function(
            self,
            input_signature=[{'input_image': tf.TensorSpec(shape=[1] + list(self.input_spec[0].shape[1:]),
                                                           dtype=tf.float32, name='image')}],
            autograph=True,
            jit_compile=True,
        )
        tf_concrete_function = tf_callable.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [tf_concrete_function], tf_callable
        )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable LiteRT ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        converter.experimental_new_converter = True
        converter.enable_resource_variables = True
        # converter.allow_custom_ops = True
        tflite_model_quant = converter.convert()

        return tflite_model_quant

    def export_tflite(self):
        quant = self.to_quant()
        model_name = "RealESRGAN"

        tflite_models_dir = pathlib.Path(f"{keras_home()}/{model_name}/tflite_models")
        os.makedirs(tflite_models_dir, exist_ok=True)
        tflite_model_quant_file = tflite_models_dir / f"RealESRGAN_x{self.upscale}_quant.tflite"
        tflite_model_quant_file.write_bytes(quant)

        logging.info("finish to export.\n"
                     f"Path : {tflite_model_quant_file}\n"
                     )
        # return tflite_model_quant_file

    def predict(
            self,
            low_resolution_image,
            batch_size=1,
            patches_size=192,
            padding = 24,
            pad_size = 15,
    ):
        """
        :param low_resolution_image: numpy array of Single Image. It is not batched.
            1. shape : [h,w,3]
            2. range : 0~255
            3. rgb channel
        :param batch_size:
        :return:
        """
        input_shape = self.input_spec[0].shape
        h, w = input_shape[1:3]
        assert h == w, "height and width must be same. (square)"
        assert patches_size+ 2*padding == h, "patches_size + 2 x padding equal to height of model input."

        low_resolution_image = pad_reflect(low_resolution_image, pad_size)
        patches, p_shape = split_image_into_overlapping_patches(low_resolution_image, patch_size=patches_size, overlap_size=padding)
        patches = patches / 255

        if self.upscale == 2:
            patches = unshuffle_patches(patches)

        num_of_patch = patches.shape[0]
        result = []
        for i in tqdm.trange(0, num_of_patch, batch_size):
            result += [self(patches[i:i + batch_size]).numpy()]

        result_concat = np.concatenate(result, axis=0)
        sr_image = result_concat.clip(0, 1)

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], self.upscale)) + (3,)
        scaled_image_shape = tuple(np.multiply(low_resolution_image.shape[0:2], self.upscale)) + (3,)
        np_sr_image = stich_together(
            sr_image, padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape, padding_size=padding * self.upscale
        )
        sr_img = (np_sr_image * 255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size * self.upscale)

        return sr_img

# def pixel_unshuffle( x: tf.Tensor, upscale):
#     """ Pixel unshuffle.
#
#     Args:
#         x (Tensor): Input feature with shape (b, c, hh, hw).
#         upscale (int): Downsample ratio.
#
#     Returns:
#         Tensor: the pixel unshuffled feature.
#     """
#     b, hh, hw, c = keras.ops.shape(x)
#     out_channel = c * (upscale**2)
#     h = hh // upscale
#     w = hw // upscale
#
#     x = keras.layers.Reshape([h, upscale, w, scale, c])(x)
#     x = keras.layers.Permute((1, 3, 5, 2, 4))(x)
#     return keras.layers.Reshape([h, w, out_channel])(x)
