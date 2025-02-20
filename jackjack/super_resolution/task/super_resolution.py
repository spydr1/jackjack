import glob
import logging

import kagglehub
import tensorflow as tf
import tensorflow.keras as keras
# import keras -- keras3
# import tf_keras -- keras2

from typing import Any, List, Optional, Tuple, Union, Dict
from official.core import task_factory
from official.core import base_task
from official.core.base_task import OptimizationConfig, RuntimeConfig, DifferentialPrivacyConfig
from official.vision.dataloaders import input_reader_factory

from jackjack.super_resolution.basicsr.legacy.v2.data.degradations import DegradationV3
from jackjack.super_resolution.basicsr.legacy.v2.data.real_esrgan_dataset import RealESRGANDataset
from jackjack.super_resolution.config.super_resolution import SuperResolutionTask, DataConfig, Degradation
from jackjack.super_resolution.drct.legacy.v2.drct import DRCT


@task_factory.register_task_cls(SuperResolutionTask)
class SuperResolutionTask(base_task.Task):
    def initialize(self, model: keras.Model):
        """Loading pretrained checkpoint."""
        if not self.task_config.init_checkpoint:
            return

        ckpt_dir_or_file = self.task_config.init_checkpoint

        if 'h5' in ckpt_dir_or_file:
            if tf.io.gfile.isdir(ckpt_dir_or_file):
                raise Exception("Please set specific file, not directory.")
            else :
                model.load_weights(ckpt_dir_or_file)

        else :
            if tf.io.gfile.isdir(ckpt_dir_or_file):
                ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

            if self.task_config.init_checkpoint_modules == 'all':
                ckpt = tf.train.Checkpoint(model=model)
                status = ckpt.read(ckpt_dir_or_file)
                status.expect_partial().assert_existing_objects_matched()
            # elif self.task_config.init_checkpoint_modules == 'backbone':
            #     ckpt = tf.train.Checkpoint(backbone=model.backbone)
            #     status = ckpt.read(ckpt_dir_or_file)
            #     status.expect_partial().assert_existing_objects_matched()

        logging.info('Finished loading pretrained checkpoint from %s',
                     ckpt_dir_or_file)

    def build_model(self) -> keras.Model:

        input_layer = keras.layers.Input(
            shape=self.task_config.model.input_shape,
            batch_size=self.task_config.train_data.global_batch_size,
            name="input_image")

        cfg = self.task_config.model.backbone.drct.as_dict()
        # if self.task_config.model.backbone.type == 'drct':

        # todo : build backbone
        drct_model = DRCT(**cfg)
        model = keras.Model(inputs=input_layer, outputs=drct_model(input_layer))

        if self.task_config.losses.perceptual_loss:
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
            rotation_range = 0.25  # todo : add to config

            def read_image(image_path):
                """
                read image and crop.
                Small image will be padded. Larger image will be cropped.

                :param image_path:
                :return:
                """
                raw = tf.io.read_file(image_path)
                image = tf.io.decode_png(raw, 3)
                # image = tf.image.random_crop(value=image, size=(h, w, 3))

                result = {"input_raw_image": image}
                return result  # , image.shape

            augmenter = keras.Sequential(
                layers=[
                    keras.layers.RandomFlip(),
                    keras.layers.RandomRotation(rotation_range),
                    keras.layers.Rescaling(scale=1.0 / 255.),
                ]
            )
            crop_layer = keras.layers.RandomCrop(h, w)

            high_h, high_w = params.target_image_shape[:2]
            low_h, low_w = high_h//params.upscale, high_w//params.upscale
            # todo
            # target_image_shape 와 cropped_image_shape를 동일하게 만들면 high에 대한 resize 필요 없음
            # target_image_shape 와 cropped_image_shape를 각각 만든 이유는 degradation 부분 때문

            # "degradation 부분 때문" -> degradation 과정에는 resize 과정이 포함되는데
            # 1. crop을 하지 않으면 너무 많은 영역에 대해 추가적인 계산을 하게되고
            # 2. 이를 해결하기 위해 너무 작은 영역에 대해 crop하면 resize 과정에서 데이터 손실이 유발됨.
            resizer_high = keras.layers.Resizing(height=high_h,width=high_w)
            resizer_low = keras.layers.Resizing(height=low_h, width=low_w)

            def crop_image(parsed_tensors: dict):
                cropped_image = crop_layer(parsed_tensors["input_raw_image"])

                parsed_tensors.update(
                    {"input_raw_image": cropped_image})
                return parsed_tensors

            def keras_augment(parsed_tensors: dict):
                parsed_tensors.update({"input_raw_image": augmenter(parsed_tensors["input_raw_image"])})
                return parsed_tensors

            def resize_image(parsed_tensors: dict):
                image = parsed_tensors["input_raw_image"]
                parsed_tensors["input_low_resolution_image"] =  resizer_low(image)
                parsed_tensors["input_high_resolution_image"] =  resizer_high(image)
                return parsed_tensors

            def prepare_dataset(
                    image_paths,
                    batch_size=1,
                    shuffle_size=4, ):

                dataset = tf.data.Dataset.from_tensor_slices((image_paths))
                dataset = dataset.repeat()
                dataset = dataset.shuffle(shuffle_size)
                dataset = dataset.map(read_image, num_parallel_calls=AUTO) # .batch(batch_size)
                dataset = dataset.map(crop_image, num_parallel_calls=AUTO).batch(batch_size)  # .batch(batch_size)
                dataset = dataset.map(keras_augment, num_parallel_calls=AUTO)  # .batch(batch_size)
                dataset = dataset.map(resize_image, num_parallel_calls=AUTO)  # .batch(batch_size)
                return dataset.prefetch(AUTO)

            dataset = prepare_dataset(training_image_paths,
                                      shuffle_size=params.shuffle_buffer_size,
                                      batch_size=params.global_batch_size)

            return dataset.repeat()
        else:
            return self.test_inputs()


    def test_inputs(self) -> tf.data.Dataset:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        dataset = tf.data.Dataset.from_tensor_slices(

            (x_train[:1000], keras.preprocessing.image.smart_resize(x_train[:1000], size=(128, 128))))

        def prepare_dict(x0, x1):
            return {"input_low_resolution_image": x0, "input_high_resolution_image": x1}

        dataset = dataset.map(prepare_dict)
        return dataset.batch(batch_size=1).repeat()

    def _build_perceptual_model(self):

        # input image for vgg, zero center, bgr
        _vgg_net = keras.applications.VGG19(
            include_top=False,
            input_shape=[
                self.task_config.model.input_shape[0] * 4,
                self.task_config.model.input_shape[1] * 4,
                self.task_config.model.input_shape[2]
            ]
        )

        # todo : add to config.
        self.perceptual_target_layer = {
            'block1_conv2': 0.1,
            'block2_conv2': 0.1,
            'block3_conv4': 1,
            'block4_conv4': 1,
            'block5_conv4': 1
        }

        self.criterion = '1'  # todo : add to config --- `'fro'`, `'euclidean'`, `1`, `2`, `np.inf`

        self.vgg = keras.Model(
            _vgg_net.input,
            [_vgg_net.get_layer(k).output for k, v in self.perceptual_target_layer.items()]
        )
        self.vgg.trainable = False

    def _build_perceptual_loss(self, high_resolution_image, model_outputs):
        """

        :param high_resolution_image: high resolution image (ground truth) - [b, w, h, 3] (rgb)
        :param model_outputs: prediction of Gan model. - [b, w, h, 3] (rgb)
        """
        # 현재 input image 범위 0~1
        model_outputs = tf.cast(model_outputs, dtype=tf.float64)
        rescaled_output = (model_outputs - 0.5 ) *2

        high_resolution_image = tf.cast(high_resolution_image, dtype=tf.float64)
        rescaled_gt = (high_resolution_image - 0.5) * 2

        #vgg input image 범위 -1~1
        x_features = self.vgg(rescaled_output)
        x_features = tf.nest.map_structure(lambda x: tf.cast(x, tf.float64), x_features)
        gt_features = self.vgg(rescaled_gt)
        gt_features = tf.nest.map_structure(lambda x: tf.cast(x, tf.float64), gt_features)

        perceptual_loss = 0.0
        layer_weight = list(self.perceptual_target_layer.values())


        for i in range(len(self.perceptual_target_layer)):
            if self.criterion == 'fro':
                perceptual_loss += tf.norm(x_features[i] - gt_features[i], ord=self.criterion) * layer_weight[i]
            elif self.criterion == '1':
                loss = keras.losses.mean_absolute_error(gt_features[i], x_features[i])
                perceptual_loss += tf.reduce_mean(loss) * layer_weight[i]
            elif self.criterion == '2':
                loss = keras.losses.mean_squared_error(gt_features[i], x_features[i])
                perceptual_loss += tf.reduce_mean(loss) * layer_weight[i]
            else :
                raise Exception(f"Criterion for perceptual loss is {self.criterion}, it is not supported.")


        return perceptual_loss

    def build_losses(self, high_resolution_image, model_outputs, aux_losses=None) -> Dict[str, tf.Tensor]:
        loss_params = self.task_config.losses
        # l1 loss
        pixel_loss = keras.losses.mean_absolute_error(high_resolution_image, model_outputs)  # todo: weight
        pixel_loss = tf.reduce_mean(pixel_loss)
        # perceptual loss
        total_loss = tf.constant(0.0, dtype=tf.float32)
        total_loss += pixel_loss * loss_params.pixel_loss_weight

        loss_result = {"pixel_loss": pixel_loss}

        if self.task_config.losses.perceptual_loss:
            perceptual_loss = self._build_perceptual_loss(high_resolution_image, model_outputs)
            loss_result.update({"perceptual_loss": perceptual_loss})
            total_loss += tf.cast(perceptual_loss * loss_params.perceptual_loss_weight, total_loss.dtype)

        loss_result.update({"total_loss": total_loss})
        return loss_result

    def build_metrics(self, training: bool = True):
        if training:
            metric_names = [
                'total_loss',
                'pixel_loss'
            ]
            if self.task_config.losses.perceptual_loss:
                metric_names.append("perceptual_loss")
            return [
                keras.metrics.Mean(name, dtype=tf.float32) for name in metric_names
            ]

    def train_step(self,
                   inputs,
                   model: keras.Model,
                   optimizer: keras.optimizers.Optimizer,
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
                high_resolution_image=high_resolution_image,
                model_outputs=outputs,
                aux_losses=model.losses)
            # Scales loss as the default gradients allreduce performs sum inside the
            # optimizer.
            scaled_loss = loss['total_loss'] / num_replicas

            # For mixed_precision policy, when LossScaleOptimizer is used, loss is
            # scaled for numerical stability.
            if isinstance(
                    optimizer, keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = optimizer.get_scaled_loss(scaled_loss)

        tvars = model.trainable_variables
        grads = tape.gradient(scaled_loss, tvars)
        # Scales back gradient before apply_gradients when LossScaleOptimizer is
        # used.
        if isinstance(
                optimizer, keras.mixed_precision.LossScaleOptimizer):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(list(zip(grads, tvars)))

        logs = {self.loss: loss["total_loss"]}
        if metrics:
            for m in metrics:
                m.update_state(loss[m.name])

        return logs

