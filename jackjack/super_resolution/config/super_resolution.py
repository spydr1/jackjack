import dataclasses
from typing import List, Optional, Union, Sequence

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.configs import common

from jackjack.super_resolution.config import backbones
from jackjack.super_resolution.legacy.drct.config import DRCT, DRCT_L


@dataclasses.dataclass
class Degradation(hyperparams.Config):
    blur_prob: float = 1.0
    resize_prob: List[float] = dataclasses.field(default_factory=lambda: [0.2, 0.7, 0.1])  # up, down, keep
    resize_range: List[float] = dataclasses.field(default_factory=lambda: [0.15, 1.5])
    gaussian_noise_prob: float = 0.5
    noise_range: List[int] = dataclasses.field(default_factory=lambda: [1, 30])
    poisson_scale_range: List[float] = dataclasses.field(default_factory=lambda: [0.05, 3])
    gray_noise_prob: float = 0.4
    jpeg_range: List[int] = dataclasses.field(default_factory=lambda: [30, 95])


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
    """Input config for training."""
    input_path: Union[Sequence[str], str, hyperparams.Config] = ''
    # todo:
    # 1. 전처리 되기전 이미지, model input size 보다 대개 raw image가 크고 어차피 crop 될 것 이므로 전체 이미지 중 일부만 사용
    # 2. 최종 target_image_shape 보다는 좀더 넉넉한 크기

    cropped_image_shape: List[int] = dataclasses.field(default_factory=list)
    upscale: int = 2
    target_image_shape: List[int] = dataclasses.field(default_factory=list)
    is_kaggle_dataset: bool = True
    weights: Optional[hyperparams.base_config.Config] = None
    global_batch_size: int = 0
    is_training: bool = True
    dtype: str = 'float32'
    shuffle_buffer_size: int = 10000
    cycle_length: int = 10
    degradations: Union[Degradation, List[Degradation], None] = None


@dataclasses.dataclass
class GanModel(hyperparams.Config):
    """The model config."""
    input_shape: List[int] = dataclasses.field(default_factory=list)
    backbone: backbones.Backbone = dataclasses.field(
        default_factory=lambda: backbones.Backbone(  # pylint: disable=g-long-lambda
            type='drct', drct=backbones.DRCT()
        )
    )
    dropout_rate: float = 0.0
    norm_activation: common.NormActivation = dataclasses.field(
        default_factory=lambda: common.NormActivation(use_sync_bn=False)
    )
    # Adds a BatchNormalization layer pre-GlobalAveragePooling in classification
    add_head_batch_norm: bool = False
    kernel_initializer: str = 'random_uniform'
    # Whether to output softmax results instead of logits.
    output_softmax: bool = False


@dataclasses.dataclass
class Losses(hyperparams.Config):
    pixel_loss_weight: float = 0.01
    # l2_weight_decay: float = 0.0

    ### perceptual_loss
    perceptual_loss: bool = False
    perceptual_loss_weight : float = 1.0
    ### perceptual_loss


    # perceptual_loss_layer_weights
    # vgg_type: vgg19
    # use_input_norm: true
    # range_norm: false
    # perceptual_weight: 1.0
    # style_weight: 0
    # criterion: l1
    # gan_opt:
    # type: GANLoss
    # gan_type: vanilla
    # real_label_val: 1.0
    # fake_label_val: 0.0
    # loss_weight: !!float
    # 5e-3
    #
    #
    # net_d_iters: 1
    # net_d_init_iters: 0


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
    top_k: int = 5
    precision_and_recall_thresholds: Optional[List[float]] = None
    report_per_class_precision_and_recall: bool = False

    # backbone: backbones.Backbone = dataclasses.field(
    #     default_factory=lambda: backbones.Backbone(
    #         type='resnet', resnet=backbones.ResNet()
    #     )
    # )


#
# @dataclasses.dataclass
# class VGGNet(hyperparams.Config):
#     model_id: str = "vgg19"
#     include_top: bool = False
#     weights: str = "imagenet"
#     pooling: Optional[bool] = None
#     classes: int = 1000
#     classifier_activation: str = "softmax"
#
#
# @dataclasses.dataclass
# class ClassificationModel(hyperparams.OneOfConfig):
#     type: Optional[str] = "vggnet"
#     vggnet: VGGNet = dataclasses.field(default_factory=VGGNet)
#
#
# @dataclasses.dataclass
# class PerceptualModel(hyperparams.Config):
#     """The model config."""
#     # input_shape: List[int] = dataclasses.field(default_factory=list)
#     # vgg_type: vgg19
#     # classification_model : ClassificationModel= dataclasses.field(
#     #   default_factory=ClassificationModel
#     # )
#
#     layer_weights: Dict = dataclasses.field(
#         default_factory=dict,
#         init={
#             'block1_conv2': 0.1,
#             'block2_conv2': 0.1,
#             'block3_conv4': 1,
#             'block4_conv4': 1,
#             'block5_conv4': 1}
#     )
#     use_input_norm: bool = True
#     perceptual_weight: float = 1.0
#     style_weight: float = 0
#     range_norm: bool = False
#     # criterion : l1

class PerceptualLoss(hyperparams.Config):
    perceptual_weight: float = 1.0


@dataclasses.dataclass
class SuperResolutionTask(cfg.TaskConfig):
    """The task config."""
    model: GanModel = dataclasses.field(
        default_factory=GanModel
    )

    # perceptual_model: PerceptualModel = dataclasses.field(
    #     default_factory=PerceptualModel
    # )

    train_data: DataConfig = dataclasses.field(
        default_factory=lambda: DataConfig(is_training=True)
    )
    validation_data: DataConfig = dataclasses.field(
        default_factory=lambda: DataConfig(is_training=False)
    )
    losses: Losses = dataclasses.field(default_factory=Losses)
    evaluation: Evaluation = dataclasses.field(default_factory=Evaluation)
    train_input_partition_dims: Optional[List[int]] = dataclasses.field(
        default_factory=list)
    eval_input_partition_dims: Optional[List[int]] = dataclasses.field(
        default_factory=list)
    init_checkpoint: Optional[str] = None
    init_checkpoint_modules: str = 'all'  # all or backbone


@exp_factory.register_config_factory('super_resolution_drct_df2k_ost')
def super_resolution_drct_df2k_ost() -> cfg.ExperimentConfig:
    steps_per_epoch = 1000

    epoch = 50
    train_batch_size = 1
    # eval_batch_size = 8
    # SuperResolutionTask
    input_shape = [64, 64, 3]
    upscale = 4

    config = cfg.ExperimentConfig(
        runtime=cfg.RuntimeConfig(
            enable_xla=False,
            # run_eagerly=True,
            mixed_precision_dtype='float16',
            num_gpus=1),
        task=SuperResolutionTask(
            model=GanModel(
                input_shape=input_shape,
                backbone=backbones.Backbone(
                    type='drct',
                    drct=DRCT_L(img_size=input_shape[0],
                                upscale=upscale),
                ),
            ),
            # losses=Losses(l2_weight_decay=0.00004),
            train_data=DataConfig(
                input_path="thaihoa1476050/df2k-ost",
                cropped_image_shape=[512, 512, 3],
                target_image_shape=[input_shape[0]*upscale, input_shape[1]*upscale, 3],
                upscale=upscale,
                shuffle_buffer_size=1000,
                is_training=True,
                global_batch_size=train_batch_size,
                degradations=[Degradation(),
                              Degradation(
                                  blur_prob=0.8,
                                  resize_prob=[0.3, 0.4, 0.3],
                                  resize_range=[0.3, 1.2],
                                  gaussian_noise_prob=0.5,
                                  noise_range=[1, 25],
                                  poisson_scale_range=[0.05, 2.5],
                                  gray_noise_prob=0.4,
                                  jpeg_range=[30, 95],

                              )]
            ),
        ),

        trainer=cfg.TrainerConfig(
            train_steps=steps_per_epoch * epoch,
            # steps_per_loop=steps_per_epoch,
            steps_per_loop=steps_per_epoch,
            summary_interval=steps_per_epoch,
            checkpoint_interval=steps_per_epoch,
            optimizer_config=optimization.OptimizationConfig({
                'optimizer': {
                    'type': 'adamw',
                    'adamw': {
                        'weight_decay_rate': 1e-2
                    }
                },
                'learning_rate': {
                    'type': 'polynomial',
                    'polynomial': {
                        'decay_steps': steps_per_epoch * epoch,
                        'initial_learning_rate': 1e-4,
                        'end_learning_rate': 1e-6,
                    }
                },
                'warmup': {
                    'type': 'linear',
                    'linear': {
                        'warmup_steps': 2000,
                        'warmup_learning_rate': 1e-7
                    }
                }
            })),
    )
    return config

@exp_factory.register_config_factory('super_resolution_drct_df2k_ost_test')
def super_resolution_drct_df2k_ost_test() -> cfg.ExperimentConfig:
    steps_per_epoch = 1000

    epoch = 10
    train_batch_size = 1
    # eval_batch_size = 8
    # SuperResolutionTask
    # todo : add scale argument
    input_shape = [64, 64, 3]
    upscale = 2

    config = cfg.ExperimentConfig(
        runtime=cfg.RuntimeConfig(
            enable_xla=False,
            # run_eagerly=True,
            mixed_precision_dtype='float16',
            num_gpus=1),
        task=SuperResolutionTask(
            model=GanModel(
                input_shape=input_shape,
                backbone=backbones.Backbone(
                    type='drct',
                    drct=DRCT(img_size=input_shape[0], upscale=upscale),
                ),
            ),
            losses=Losses(l2_weight_decay=0.00004),
            train_data=DataConfig(
                input_path="thaihoa1476050/df2k-ost",
                cropped_image_shape=[512, 512, 3],
                target_image_shape=[input_shape[0]*upscale, input_shape[1]*upscale, 3],
                upscale=upscale,
                shuffle_buffer_size=100,
                is_training=True,
                global_batch_size=train_batch_size,
                degradations=[Degradation(),
                              Degradation(
                                  blur_prob=0.8,
                                  resize_prob=[0.3, 0.4, 0.3],
                                  resize_range=[0.3, 1.2],
                                  gaussian_noise_prob=0.5,
                                  noise_range=[1, 25],
                                  poisson_scale_range=[0.05, 2.5],
                                  gray_noise_prob=0.4,
                                  jpeg_range=[30, 95],

                              )]
            ),
        ),

        trainer=cfg.TrainerConfig(
            train_steps=steps_per_epoch * epoch,
            # steps_per_loop=steps_per_epoch,
            steps_per_loop=10,
            summary_interval=steps_per_epoch,
            checkpoint_interval=steps_per_epoch,
            optimizer_config=optimization.OptimizationConfig({
                'optimizer': {
                    'type': 'adamw',
                    'adamw': {
                        'weight_decay_rate': 1e-2
                    }
                },
                'learning_rate': {
                    'type': 'polynomial',
                    'polynomial': {
                        'decay_steps': steps_per_epoch * epoch,
                        'initial_learning_rate': 1e-4,
                        'end_learning_rate': 1e-6,
                    }
                },
                'warmup': {
                    'type': 'linear',
                    'linear': {
                        'warmup_steps': 2000,
                        'warmup_learning_rate': 1e-7
                    }
                }
            })),
    )
    return config

@exp_factory.register_config_factory('super_resolution_perceptual_loss_test')
def perceptual_test() -> cfg.ExperimentConfig:
    steps_per_epoch = 1000

    epoch = 50
    train_batch_size = 1
    # eval_batch_size = 8
    # SuperResolutionTask
    input_shape = [64, 64, 3]
    upscale = 4

    config = cfg.ExperimentConfig(
        runtime=cfg.RuntimeConfig(
            enable_xla=False,
            # run_eagerly=True,
            mixed_precision_dtype='float16',
            num_gpus=1),
        task=SuperResolutionTask(
            model=GanModel(
                input_shape=input_shape,
                backbone=backbones.Backbone(
                    type='drct',
                    drct=DRCT_L(img_size=input_shape[0],
                              upscale=upscale),
                ),
            ),
            init_checkpoint='/home/data/tensorflow/drct/kaggle/drct_030_0.03757117688655853.weights.h5',
            losses=Losses(perceptual_loss=True),
            train_data=DataConfig(
                input_path="thaihoa1476050/df2k-ost",
                cropped_image_shape=[512, 512, 3],
                target_image_shape=[input_shape[0]*upscale, input_shape[1]*upscale, 3],
                upscale=upscale,
                shuffle_buffer_size=1000,
                is_training=True,
                global_batch_size=train_batch_size,
            ),
        ),

        trainer=cfg.TrainerConfig(
            train_steps=steps_per_epoch * epoch,
            # steps_per_loop=steps_per_epoch,
            steps_per_loop=100,
            summary_interval=100,
            checkpoint_interval=steps_per_epoch,
            optimizer_config=optimization.OptimizationConfig({
                'optimizer': {
                    'type': 'adamw',
                    'adamw': {
                        'weight_decay_rate': 1e-2
                    }
                },
                'learning_rate': {
                    'type': 'polynomial',
                    'polynomial': {
                        'decay_steps': steps_per_epoch * epoch,
                        'initial_learning_rate': 1e-4,
                        'end_learning_rate': 1e-6,
                    }
                },
                'warmup': {
                    'type': 'linear',
                    'linear': {
                        'warmup_steps': 2000,
                        'warmup_learning_rate': 1e-7
                    }
                }
            })),
    )
    return config