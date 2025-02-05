import glob
import logging

import kagglehub
import tensorflow as tf
import tf_keras
from typing import Any, List, Optional, Tuple, Union
from official.core import task_factory
from official.core import base_task
from official.core.base_task import OptimizationConfig, RuntimeConfig, DifferentialPrivacyConfig

from jackjack.super_resolution.basicsr.legacy.v2.data.degradations import DegradationV3
from jackjack.super_resolution.basicsr.legacy.v2.data.real_esrgan_dataset import RealESRGANDataset
from jackjack.super_resolution.config.super_resolution import SuperResolutionTask, DataConfig, Degradation
from jackjack.super_resolution.drct.legacy.v2.drct import DRCT


@task_factory.register_task_cls(SuperResolutionTask)
class SuperResolutionTask(base_task.Task):
    def initialize(self, model: tf_keras.Model):
        """Loading pretrained checkpoint."""
        if not self.task_config.init_checkpoint:
            return

        ckpt_dir_or_file = self.task_config.init_checkpoint
        if tf.io.gfile.isdir(ckpt_dir_or_file):
            ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

        # Restoring checkpoint.
        if self.task_config.init_checkpoint_modules == 'all':
            ckpt = tf.train.Checkpoint(**model.checkpoint_items)
            status = ckpt.read(ckpt_dir_or_file)
            status.expect_partial().assert_existing_objects_matched()
        # else:
        #     ckpt_items = {}
        #     if 'backbone' in self.task_config.init_checkpoint_modules:
        #         ckpt_items.update(backbone=model.backbone)
        #     if 'decoder' in self.task_config.init_checkpoint_modules:
        #         ckpt_items.update(decoder=model.decoder)
        #
        #     ckpt = tf.train.Checkpoint(**ckpt_items)
        #     status = ckpt.read(ckpt_dir_or_file)
        #     status.expect_partial().assert_existing_objects_matched()

        logging.info('Finished loading pretrained checkpoint from %s',
                     ckpt_dir_or_file)

    def build_model(self) -> tf_keras.Model:

        input_layer = tf_keras.layers.Input(
            shape=self.task_config.model.input_shape,
            batch_size=self.task_config.train_data.global_batch_size,
            name="input_image")

        cfg = self.task_config.model.backbone.drct.as_dict()
        # if self.task_config.model.backbone.type == 'drct':

        drct_model = DRCT(**cfg)
        model = tf_keras.Model(inputs=input_layer, outputs=drct_model(input_layer))

        if self.task_config.perceptual_loss:
            self._build_perceptual_model()

        return model

    def build_inputs(self,
                     params: DataConfig,
                     input_context: Optional[tf.distribute.InputContext] = None):

        AUTO = tf.data.AUTOTUNE
        if params.is_kaggle_dataset and params.is_training and params.input_path != '':
            path = kagglehub.dataset_download(params.input_path)
            training_image_paths = glob.glob(f"{path}/train/**/*.png", recursive=True)

            h, w, _ = params.cropped_image_shape

            assert h == w, "Input must be square sized image."
            rotation_range = 0.25  # todo

            def read_image(image_path):
                """
                read image and crop.
                Small image will be padded. Larger image will be cropped.

                :param image_path:
                :return:
                """
                raw = tf.io.read_file(image_path)
                image = tf.io.decode_png(raw, 3)
                result = {"input_raw_image": image}
                return result  # , image.shape

            augmenter = tf_keras.Sequential(
                layers=[
                    tf_keras.layers.RandomFlip(),
                    tf_keras.layers.RandomRotation(rotation_range),
                    tf_keras.layers.Rescaling(scale=1.0 / 255.),
                ]
            )
            with tf.device("cpu"):
                crop_layer = tf_keras.layers.RandomCrop(h, w)

            kernel_layer = self.get_kernel_layer()
            degradation_layer: DegradationV3 = self.get_degradation_layer(
                target_image_shape=params.target_image_shape,  # todo
                params=params,
            )

            def crop_image(parsed_tensors: dict):
                with tf.device("cpu"):
                    cropped_image = crop_layer(parsed_tensors["input_raw_image"])

                parsed_tensors.update(
                    {"input_raw_image": cropped_image})
                return parsed_tensors

            def keras_augment(parsed_tensors: dict):
                parsed_tensors.update({"input_raw_image": augmenter(parsed_tensors["input_raw_image"])})
                return parsed_tensors

            def keras_degradation(parsed_tensors: dict):
                gpus = tf.config.list_physical_devices('GPU')
                tpus = tf.config.list_physical_devices('TPU')

                if tpus:
                    device = tf.device("tpu")
                elif gpus:
                    device = tf.device("gpu")
                else :
                    device = tf.device("cpu")

                # todo : with cpu, depthwise_conv2d 엄청 느려짐 .. 버그일까 ?

                with device:
                    degradation_result = degradation_layer(parsed_tensors["input_raw_image"],
                                                           kernel1=parsed_tensors["kernel1"],
                                                           kernel2=parsed_tensors["kernel2"],
                                                           sinc_kernel=parsed_tensors["sinc_kernel"],
                                                           )
                parsed_tensors.update(
                    degradation_result
                )
                return parsed_tensors

            def get_kernel(parsed_tensors):
                parsed_tensors.update(**kernel_layer.get_kernel())
                return parsed_tensors

            def prepare_dataset(
                    image_paths,
                    batch_size=1,
                    shuffle_size=4, ):
                dataset = tf.data.Dataset.from_tensor_slices((image_paths))
                dataset = dataset.shuffle(shuffle_size)
                dataset = dataset.map(read_image, num_parallel_calls=AUTO)  # .batch(batch_size)
                dataset = dataset.map(crop_image, num_parallel_calls=AUTO).batch(batch_size)  # .batch(batch_size)
                dataset = dataset.map(keras_augment, num_parallel_calls=AUTO)  # .batch(batch_size)
                dataset = dataset.map(get_kernel, num_parallel_calls=AUTO)  # .batch(batch_size)

                dataset = dataset.map(keras_degradation, num_parallel_calls=AUTO)  # .batch(batch_size)
                # dataset = dataset.map(prepare_dict, num_parallel_calls=AUTO)
                return dataset.prefetch(AUTO)

            dataset = prepare_dataset(training_image_paths,
                                      shuffle_size=params.shuffle_buffer_size,
                                      batch_size=params.global_batch_size)

            return dataset.repeat()

        else:
            return self.test_inputs()

    def get_kernel_layer(self):
        kernel_layer = RealESRGANDataset(
            blur_kernel_size=21,
            kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso',
                         'plateau_aniso'],
            kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            sinc_prob=0.1,
            blur_sigma=[0.2, 3],
            betag_range=[0.5, 4],
            betap_range=[1, 2],
            blur_kernel_size2=21,
            kernel_list2=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso',
                          'plateau_aniso'],
            kernel_prob2=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            sinc_prob2=0.1,
            blur_sigma2=[0.2, 1.5],
            betag_range2=[0.5, 4],
            betap_range2=[1, 2],
            final_sinc_prob=0.8,
        )
        return kernel_layer

    def get_degradation_layer(self, target_image_shape, params: DataConfig,
                              gt_usm=True, seed=12345, ):
        # degradation_layer = keras.layers.Identity()
        # if params is None :
        #     pass
        # elif isinstance(params,Degradation):
        #     # todo
        #     pass
        degradation_cfg = params.degradations
        if isinstance(degradation_cfg, list):
            assert len(degradation_cfg) == 2, "number of degradation step is limit to 2."
            # first, second
            _f, _s = degradation_cfg[0], degradation_cfg[1]

            degradation_layer = DegradationV3(
                resize_prob=_f.resize_prob,
                resize_prob2=_s.resize_prob,
                resize_range=_f.resize_range,
                resize_range2=_s.resize_range,
                gray_noise_prob=_f.gray_noise_prob,
                gray_noise_prob2=_s.gray_noise_prob,
                gaussian_noise_prob=_f.gaussian_noise_prob,
                gaussian_noise_prob2=_s.gaussian_noise_prob,
                noise_range=_f.noise_range,
                noise_range2=_s.noise_range,
                poisson_scale_range=_f.poisson_scale_range,
                poisson_scale_range2=_s.poisson_scale_range,
                jpeg_range=_f.jpeg_range,
                jpeg_range2=_s.jpeg_range,
                second_blur_prob=_s.blur_prob,
                patch_size=target_image_shape[0],
                seed=seed,
                gt_usm=gt_usm,
                batch_size=params.global_batch_size,
                scale=params.upscale
            )
        else:
            degradation_layer = tf_keras.layers.Identity()

        return degradation_layer

    def test_inputs(self) -> tf.data.Dataset:
        (x_train, y_train), (x_test, y_test) = tf_keras.datasets.cifar10.load_data()
        dataset = tf.data.Dataset.from_tensor_slices(

            (x_train[:1000], tf_keras.preprocessing.image.smart_resize(x_train[:1000], size=(128, 128))))

        def prepare_dict(x0, x1):
            return {"input_low_resolution_image": x0, "input_high_resolution_image": x1}

        dataset = dataset.map(prepare_dict)
        return dataset.batch(batch_size=1).repeat()

    def _build_perceptual_model(self):

        _vgg_net = tf_keras.applications.VGG19(
            include_top=False,
            input_shape=[
                self.task_config.model.input_shape[0] * 4,
                self.task_config.model.input_shape[1] * 4,
                self.task_config.model.input_shape[2]
            ]
        )
        self.perceptual_target_layer = {
            'block1_conv2': 0.1,
            'block2_conv2': 0.1,
            'block3_conv4': 1,
            'block4_conv4': 1,
            'block5_conv4': 1
        }
        self.vgg = tf_keras.Model(
            _vgg_net.input,
            [_vgg_net.get_layer(k).output for k, v in self.perceptual_target_layer.items()]
        )

    def _build_perceptual_loss(self, high_resolution_image, model_outputs):
        """

        :param high_resolution_image: high resolution image (ground truth) - [b, w, h, 3](rgb)
        :param model_outputs: prediction of Gan model.

        :return: perceptual_loss
        """
        prep = tf_keras.applications.vgg19.preprocess_input

        x_features = self.vgg(prep(model_outputs))
        gt_features = self.vgg(prep(high_resolution_image))

        perceptual_loss = 0.0
        layer_weight = list(self.perceptual_target_layer.values())
        for i in range(len(self.perceptual_target_layer)):
            tf_keras.losses.MeanAbsoluteError()
            perceptual_loss += tf_keras.losses.MeanAbsoluteError()(gt_features[i], x_features[i],
                                                                   sample_weight=layer_weight[i])
            # keras.ops.mean(keras.losses.mean_absolute_error(gt_features[i], x_features[i])*layer_weight[i])

        return perceptual_loss

    def build_losses(self, high_resolution_image, model_outputs, aux_losses=None) -> tf.Tensor:

        # l1 loss
        pixel_loss = tf_keras.losses.mean_absolute_error(high_resolution_image, model_outputs)  # todo: weight
        pixel_loss = tf.reduce_mean(pixel_loss)
        # perceptual loss
        total_loss = 0.0
        total_loss += pixel_loss
        if self.task_config.perceptual_loss:
            perceptual_loss = self._build_perceptual_loss(high_resolution_image, model_outputs)
            total_loss += perceptual_loss

        loss_result = {"total_loss": total_loss,
                       "pixel_loss": pixel_loss,
                       }
        return loss_result

    def build_metrics(self, training: bool = True):
        if training:
            metric_names = [
                'total_loss',
                'pixel_loss'
            ]
            if self.task_config.perceptual_loss:
                metric_names.append("perceptual_loss")
            return [
                tf_keras.metrics.Mean(name, dtype=tf.float32) for name in metric_names
            ]

    def train_step(self,
                   inputs,
                   model: tf_keras.Model,
                   optimizer: tf_keras.optimizers.Optimizer,
                   metrics=None):
        """

        :param inputs:
            dictionary
                input_low_resolution_image : [batch, width, height, 3]
                input_high_resolution_image : [batch, width * scale, height * scale, 3]
        :param model:
        :param optimizer:
        :param metrics:
        :return:
        """

        low_resolution_image = inputs["input_low_resolution_image"]
        high_resolution_image = inputs["input_high_resolution_image"]

        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        with tf.GradientTape() as tape:
            outputs = model(low_resolution_image, training=True)

            # Casting output layer as float32 is necessary when mixed_precision is
            # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
            outputs = tf.nest.map_structure(
                lambda x: tf.cast(x, tf.float32), outputs)

            # Computes per-replica loss.
            loss = self.build_losses(
                model_outputs=outputs,
                high_resolution_image=high_resolution_image,
                aux_losses=model.losses)
            # Scales loss as the default gradients allreduce performs sum inside the
            # optimizer.
            scaled_loss = loss['total_loss'] / num_replicas

            # For mixed_precision policy, when LossScaleOptimizer is used, loss is
            # scaled for numerical stability.
            if isinstance(
                    optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = optimizer.get_scaled_loss(scaled_loss)

        tvars = model.trainable_variables
        grads = tape.gradient(scaled_loss, tvars)
        # Scales back gradient before apply_gradients when LossScaleOptimizer is
        # used.
        if isinstance(
                optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(list(zip(grads, tvars)))

        logs = {self.loss: loss["total_loss"]}
        if metrics:
            for m in metrics:
                m.update_state(loss[m.name])

        return logs

    # def build_inputs(self,
    #                params,
    #                input_context: Optional[tf.distribute.InputContext] = None):


def pad_to_target_size(
        image,
        target_height,
        target_width,
        mode='REFLECT',
        constant_values=0,
        name=None
):
    # https://www.tensorflow.org/api_docs/python/tf/pad
    """

    :param image: [h,w,c]
    :param target_height:
    :param target_width:
    :param mode:
    :param constant_values:
    :param name:
    :return:
        [target_height, target_width, c]. case 1. If original image is less than target size.
        [h, w, c].                        case 2. If original image is larger than target size.
    """

    shape = tf.shape(image)
    h, w = shape[0], shape[1]

    pad_h = tf.maximum(0, target_height - h)
    pad_w = tf.maximum(0, target_width - w)
    # padding = keras.ops.repeat(tf.constant([[0,pad_h], [0,pad_w]])[:,:,None], repeats=3, axis=-1)
    padded_image = tf.pad(image, [[0, pad_h], [0, pad_w], [0, 0]], mode=mode)
    tf_keras.preprocessing.image.smart_resize
    return padded_image
