import dataclasses
from typing import List, Optional, Sequence, Union

from official.modeling import hyperparams


@dataclasses.dataclass
class DRCT(hyperparams.Config):
    img_size: int = 64
    patch_size: int = 1
    in_chans: int = 3
    embed_dim: int = 180
    depths: List[int] = dataclasses.field(
        default_factory=lambda: [6, 6, 6, 6, 6, 6])
    # [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    num_heads: List[int] = dataclasses.field(
        default_factory=lambda: [6, 6, 6, 6, 6, 6])
    window_size: int = 16
    compress_ratio: int = 3
    squeeze_factor: int = 30
    conv_scale: float = 0.01
    overlap_ratio: float = 0.5
    mlp_ratio: float = 2.
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.1
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    upscale: int = 4
    img_range: float = 1.
    upsampler: str = 'pixelshuffle'
    resi_connection: str = '1conv'
    gc: int = 32


@dataclasses.dataclass
class DRCT_L(DRCT):
    depths: List[int] = dataclasses.field(
        default_factory=lambda: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
    # [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    num_heads: List[int] = dataclasses.field(
        default_factory=lambda: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])


WEIGHT_PATH = {
    "drct_l": {"config":DRCT_L(), "path":"spydr1/drct/keras/drct_l"},
    "4xrealwebphoto_v4_drct-l": {"config": DRCT_L(), "path": "spydr1/drct/keras/4xrealwebphoto_v4_drct-l"}
}
