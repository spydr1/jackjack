import dataclasses
from typing import List, Optional, Sequence, Union

from official.modeling import hyperparams


@dataclasses.dataclass
class RRDBNET(hyperparams.Config):
    img_width: int = 64
    img_height: int = 64
    hidden_dim : int = 64
    num_block: int = 23
    body_hidden_dim:int = 32
    upscale: int = 4

WEIGHT_PATH = {

        "x2": {"config": RRDBNET(upscale=2), "path": "spydr1/realesrgan/keras/x2"} ,
        "x4": {"config": RRDBNET(upscale=4), "path": "spydr1/realesrgan/keras/x4"},
        "x8": {"config": RRDBNET(upscale=8), "path": "spydr1/realesrgan/keras/x8"},
}

# if load_weights:
#     logging.info('weight loading.')
#     path = {
#         "x2": "spydr1/realesrgan/keras/x2",
#         "x4": "spydr1/realesrgan/keras/x4",
#         "x8": "spydr1/realesrgan/keras/x8",
#     }
#     weight_path = kagglehub.model_download(path[scale])
#     self.load_weights(f"{weight_path}/RealESRGAN_x{scale}.weights.h5")