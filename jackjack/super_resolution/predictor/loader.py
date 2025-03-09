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
    pretrained_models = _DRCT_weight_path.keys()

    @classmethod
    def info(cls):
        return print(f"Pretrained wegiht list\n",
                     "\n".join([f"{i+1}. {name}" for (i, name) in enumerate(DRCT.pretrained_models)]),
                     f"\nIt can be used for args of \"get_pretrained_model\" function.",
                     sep="")

    @staticmethod
    def get_pretrained_model(key, size=64) -> _DRCT :


        config: hyperparams.Config = _DRCT_weight_path[key]["config"]
        # todo : 가능한 사이즈 값이 뭐지 ?
        if size is not None :
            config.override({"img_size":size, "trainable":False, "fixed_shape":True}, is_strict=False)
            z = tf.random.normal([1, size, size, 3])
        else :
            logging.info("*** dynamic shape prediction *** \n"
                         "It is not recommended. Fixed size has more speed")
            z = tf.random.normal([1,64,64,3])
        path = _DRCT_weight_path[key]["path"]
        current = time.time()
        model: keras.Model= _DRCT(**config.as_dict())
        model(z) # call 해주어야 build 완료.
        # model.build([None,None,None,3]) # [b, h, w, 3]
        logging.info(f"model build : {time.time() - current}s")

        kaggle_path = kagglehub.model_download(path)
        current = time.time()
        model.load_weights(f"{kaggle_path}/generator.weights.h5")
        logging.info(f"model load : {time.time() - current}s")
        return model

    # compare to origin code
    # INFO:root:model build : 0.48326992988586426s
    # INFO:root:model load : 7.2102272510528564s


class RealESRGAN:
    pretrained_models = _RRDBNet_weight_path.keys()

    @staticmethod
    def get_pretrained_model(key, size) -> _RRDBNet:
        config: hyperparams.Config = _RRDBNet_weight_path[key]["config"]
        config.override({"img_height": size})
        config.override({"img_width": size})
        path = _RRDBNet_weight_path[key]["path"]
        model: keras.Model = _RRDBNet(**config.as_dict())
        kaggle_path = kagglehub.model_download(path)
        model.load_weights(f"{kaggle_path}/generator.weights.h5")

        return model

