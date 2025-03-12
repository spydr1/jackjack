import logging
import time
from typing import List

import kagglehub
import tensorflow as tf
import tensorflow.keras as keras

from official.modeling import hyperparams

from jackjack.super_resolution.legacy.drct.drct import DRCT as _DRCT
# from jackjack.super_resolution.legacy.drct.drctV2 import DRCT as _DRCT
from jackjack.super_resolution.legacy.drct.config import WEIGHT_PATH as _DRCT_weight_path
from jackjack.super_resolution.legacy.real_esrgan.rrdbnet import RRDBNet as _RRDBNet
from jackjack.super_resolution.legacy.real_esrgan.config import WEIGHT_PATH as _RRDBNet_weight_path

# import tf_keras as keras

logging.basicConfig(level=logging.INFO)

#todo : staticmethod 가 필요할까 ..
class DRCT:

    # todo : kaggle 에서 key 를 가져오는 것도 필요 할 수 있다.
    pretrained_models = list(_DRCT_weight_path.keys())
    @classmethod
    def info(cls):
        return print(f"Pretrained wegiht list\n",
                     "\n".join([f"{i+1}. {name}" for (i, name) in enumerate(cls.pretrained_models)]),
                     f"\nIt can be used for args of \"get_pretrained_model\" function.",
                     sep="")

    @staticmethod
    def get_pretrained_model(key = pretrained_models[0], size=64, weight_path=None) -> _DRCT :
        config: hyperparams.Config = _DRCT_weight_path[key]["config"]
        path = _DRCT_weight_path[key]["path"]

        current = time.time()
        model: keras.Model= _DRCT(input_specs = keras.layers.InputSpec(shape=[None, size, size, 3]), **config.as_dict())
        logging.info(f"model build : {time.time() - current}s")

        current = time.time()
        if weight_path:
            model.load_weights(weight_path)
        else:
            kaggle_path = kagglehub.model_download(path)
            model.load_weights(f"{kaggle_path}/generator.weights.h5")
        logging.info(f"model load : {time.time() - current}s")
        return model

    # compare to origin code
    # INFO:root:model build : 0.48326992988586426s
    # INFO:root:model load : 7.2102272510528564s

class RealESRGAN:

    pretrained_models = list(_RRDBNet_weight_path.keys())

    @classmethod
    def info(cls):
        return print(f"Pretrained wegiht list\n",
                     "\n".join([f"{i+1}. {name}" for (i, name) in enumerate(cls.pretrained_models)]),
                     f"\nIt can be used for args of \"get_pretrained_model\" function.",
                     sep="")

    @staticmethod
    def get_pretrained_model(key = pretrained_models[0], size=240, weight_path=None) -> _RRDBNet:
        config: hyperparams.Config = _RRDBNet_weight_path[key]["config"]

        # depth_to_space
        if (config.upscale == 2) :
            size*=2

        config.override({"img_height": size})
        config.override({"img_width": size})
        path = _RRDBNet_weight_path[key]["path"]
        current = time.time()
        model: keras.Model = _RRDBNet(**config.as_dict())
        logging.info(f"model build : {time.time() - current}s")

        current = time.time()
        if weight_path:
            model.load_weights(weight_path)
        else:
            kaggle_path = kagglehub.model_download(path)
            model.load_weights(f"{kaggle_path}/generator.weights.h5")
        logging.info(f"model load : {time.time() - current}s")
        return model

