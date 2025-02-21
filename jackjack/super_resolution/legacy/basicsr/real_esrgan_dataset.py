"""
Apache License Version 2.0
https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/realesrgan_dataset.py
"""

import math
import random
import numpy as np
import tensorflow as tf

from jackjack.super_resolution.legacy.basicsr.degradations import circular_lowpass_kernel, random_mixed_kernels


class RealESRGANDataset:
    def __init__(
            self,
            blur_kernel_size,
            kernel_list,
            kernel_prob,
            blur_sigma,
            betag_range,
            betap_range,
            sinc_prob,
            blur_kernel_size2,
            kernel_list2,
            kernel_prob2,
            blur_sigma2,
            betag_range2,
            betap_range2,
            sinc_prob2,
            final_sinc_prob
    ):
        self._blur_kernel_size = blur_kernel_size
        self._kernel_list = kernel_list
        self._kernel_prob = kernel_prob  # a list for each kernel probability
        self._blur_sigma = blur_sigma
        self._betag_range = betag_range  # betag used in generalized Gaussian blur kernels
        self._betap_range = betap_range  # betap used in plateau blur kernels
        self._sinc_prob = sinc_prob  # the probability for sinc filters

        # blur settings for the second degradation
        self._blur_kernel_size2 = blur_kernel_size2
        self._kernel_list2 = kernel_list2
        self._kernel_prob2 = kernel_prob2
        self._blur_sigma2 = blur_sigma2
        self._betag_range2 = betag_range2
        self._betap_range2 = betap_range2
        self._sinc_prob2 = sinc_prob2

        # a final sinc filter
        self._final_sinc_prob = final_sinc_prob

        self._kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self._pulse_tensor = np.zeros([21, 21],
                                      dtype=np.float32)  # convolving with pulse tensor brings no blurry effect
        self._pulse_tensor[10, 10] = 1

    def get_kernel(self):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self._kernel_range)
        if np.random.uniform() < self._sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self._kernel_list,
                self._kernel_prob,
                kernel_size,
                self._blur_sigma,
                self._blur_sigma, [-math.pi, math.pi],
                self._betag_range,
                self._betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self._kernel_range)
        if np.random.uniform() < self._sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self._kernel_list2,
                self._kernel_prob2,
                kernel_size,
                self._blur_sigma2,
                self._blur_sigma2, [-math.pi, math.pi],
                self._betag_range2,
                self._betap_range2,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self._final_sinc_prob:
            kernel_size = random.choice(self._kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        else:
            sinc_kernel = self._pulse_tensor

        return_d = {'kernel1': tf.constant(kernel, dtype=tf.float32), 'kernel2': tf.constant(kernel2, dtype=tf.float32),
                    'sinc_kernel': tf.constant(sinc_kernel, dtype=tf.float32)}
        return return_d
