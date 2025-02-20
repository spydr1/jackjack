import math
import tensorflow as tf
import tensorflow.keras as keras
# import tf_keras as keras
import numpy as np
# import keras.mixed_precision

def drop_path(x, drop_prob: float = 0.):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    keep_prob = 1 - drop_prob
    input_shape = tf.shape(x)
    b, l, c = input_shape[0], input_shape[1], input_shape[2]
    shape = (b,) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets

    random_tensor = keep_prob + tf.random.normal(shape, dtype=x.dtype)
    random_tensor = tf.floor(random_tensor)
    # random_tensor.floor_()  # binarize
    output = tf.divide(x, keep_prob) * random_tensor
    output = tf.cast(output, dtype=x.dtype)
    return output


class DropPath(keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, rate=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.rate = rate

    def call(self, x, training=None):
        if training and self.rate > 0:  # and self.rate > 0:
            return drop_path(x, self.rate)
        return x


## Channel Attention (CA) Layer
class ChannelAttention(keras.layers.Layer):
    """Channel attention used in RCAN.
    https://github.com/yulunzhang/RCAN/blob/3339ebc59519c3bb2b5719b87dd36515ec7f3ba7/RCAN_TrainCode/code/model/rcan.py#L9

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)

        # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
        self.attention = keras.Sequential(
            [
                keras.layers.GlobalAvgPool2D(),
                keras.layers.Conv2D(num_feat // squeeze_factor, kernel_size=1, padding='valid'),
                keras.layers.Activation(keras.activations.relu),  # inplace=True
                keras.layers.Conv2D(num_feat, kernel_size=1, padding='valid'),
                keras.layers.Activation(keras.activations.sigmoid),
            ]
        )

    def call(self, x):
        y = self.attention(x)
        return x * y


## Residual Channel Attention Block (RCAB)
# https://github.com/yulunzhang/RCAN/blob/3339ebc59519c3bb2b5719b87dd36515ec7f3ba7/RCAN_TrainCode/code/model/rcan.py#L28
class RCAB(keras.layers.Layer):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30, **kwargs):
        super(RCAB, self).__init__(**kwargs)

        self.cab = keras.Sequential(
            keras.layers.Conv2D(num_feat // compress_ratio, kernel_size=3, stride=1, padding='same'),
            keras.layers.Activation(keras.activations.gelu),
            keras.layers.Conv2D(num_feat, kernel_size=3, stride=1, padding='same'),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def call(self, x):
        return self.cab(x)


class Mlp(keras.layers.Layer):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=keras.activations.gelu, drop=0.,
                 **kwargs):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = keras.layers.Dense(hidden_features, name='fc1')
        self.act = keras.layers.Activation(act_layer)
        self.fc2 = keras.layers.Dense(out_features, name='fc2')
        self.drop = keras.layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size
#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = np.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
#     windows = np.transpose(x, axes=(0, 1, 3, 2, 4, 5))
#     windows = np.reshape(windows, [-1, window_size, window_size, C])
#     return windows


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    b, h, w, c = np.shape(x)
    x = np.reshape(x, [b, h // window_size, window_size, w // window_size, window_size, c])
    windows = np.transpose(x, axes=(0, 1, 3, 2, 4, 5))
    windows = np.reshape(windows, [-1, window_size, window_size, c])
    return windows


def window_partition_layer(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    input_shape = tf.shape(x)
    b, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    x = tf.reshape(x, [b, h // window_size, window_size, w // window_size, window_size, c])
    windows = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    windows = tf.reshape(windows, [-1, window_size, window_size, c])
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    input_shape = tf.shape(windows)
    b, hh, hw, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

    x = tf.reshape(windows, [-1, H // window_size, W // window_size, window_size, window_size, c])
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, [-1, H, W, c])
    return x


# x = keras.random.normal([1*4,128,128,3])


class WindowAttention(keras.layers.Layer):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., **kwargs):

        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # self.relative_position_bias_table = self.add_variable(
        #     ((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads),
        #     initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        #     # name="relative_position_bias_table"
        # )  # 2*Wh-1 * 2*Ww-1, nH

        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing="ij"))

        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
        coords_flatten = np.reshape(coords, [2, -1])  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = np.transpose(relative_coords, [1, 2, 0])  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = np.sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Ww

        self.qkv = keras.layers.Dense(dim * 3, use_bias=qkv_bias, name='qkv')
        self.attn_drop = keras.layers.Dropout(attn_drop)
        self.proj = keras.layers.Dense(dim, name='proj')

        self.proj_drop = keras.layers.Dropout(proj_drop)

        self.softmax = keras.layers.Softmax(axis=-1)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="relative_position_bias_table"
        )  # 2*Wh-1 * 2*Ww-1, nH

    def call(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        input_shape = tf.shape(x)
        b, n, c = input_shape[0], input_shape[1], input_shape[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [b, n, 3, self.num_heads, c // self.num_heads])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # k = keras.ops.swapaxes(k, -2, -1)
        k = tf.transpose(k, perm=[0, 1, 3, 2])
        attn = tf.matmul(q, k)

        idx = tf.reshape(self.relative_position_index, [-1])

        relative_position_bias = tf.reshape(tf.gather(self.relative_position_bias_table, idx),
                                            [self.window_size[0] * self.window_size[1],
                                             self.window_size[0] * self.window_size[1], -1])  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias[None]

        if mask is not None:
            input_shape = tf.shape(mask)
            nW = input_shape[0]
            attn = tf.reshape(attn, [-1, nW, self.num_heads, n, n]) + mask[:, None][None]
            attn = tf.reshape(attn, [-1, self.num_heads, n, n])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [b, n, c])
        x = self.proj(x)  # z = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class RDG(keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, qk_scale,
                 drop, attn_drop, drop_path, norm_layer, gc, patch_size, img_size, **kwargs):
        super(RDG, self).__init__(**kwargs)

        self.swin1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                          num_heads=num_heads, window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust1 = keras.layers.Conv2D(gc, kernel_size=1, name='adjust1')

        self.swin2 = SwinTransformerBlock(dim + gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + gc) % num_heads), window_size=window_size,
                                          shift_size=window_size // 2,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust2 = keras.layers.Conv2D(gc, kernel_size=1, name='adjust2')

        self.swin3 = SwinTransformerBlock(dim + 2 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 2 * gc) % num_heads), window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust3 = keras.layers.Conv2D(gc, kernel_size=1, name='adjust3')

        self.swin4 = SwinTransformerBlock(dim + 3 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 3 * gc) % num_heads), window_size=window_size,
                                          shift_size=window_size // 2,  # For first block
                                          mlp_ratio=1,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust4 = keras.layers.Conv2D(gc, kernel_size=1, name='adjust4')

        self.swin5 = SwinTransformerBlock(dim + 4 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 4 * gc) % num_heads), window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=1,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust5 = keras.layers.Conv2D(dim, kernel_size=1, name='adjust5')

        self.lrelu = keras.layers.LeakyReLU(alpha=0.2)

        ### no weight.
        self.pe = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.pue = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)
        ### no weight.

    def call(self, x, x_size=None, training=False):
        x1 = self.pe(self.lrelu(self.adjust1(self.pue(self.swin1(x, x_size=x_size, training=training), x_size=x_size))))
        x2 = self.pe(self.lrelu(self.adjust2(
            self.pue(self.swin2(tf.concat((x, x1), -1), x_size=x_size, training=training), x_size=x_size))))
        x3 = self.pe(self.lrelu(self.adjust3(
            self.pue(self.swin3(tf.concat((x, x1, x2), -1), x_size=x_size, training=training), x_size=x_size))))
        x4 = self.pe(self.lrelu(self.adjust4(
            self.pue(self.swin4(tf.concat((x, x1, x2, x3), -1), x_size=x_size, training=training), x_size=x_size))))
        x5 = self.pe(self.adjust5(
            self.pue(self.swin5(tf.concat((x, x1, x2, x3, x4), -1), x_size=x_size, training=training), x_size=x_size)))

        return x5 * 0.2 + x


class SwinTransformerBlock(keras.layers.Layer):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=keras.activations.gelu, norm_layer=keras.layers.LayerNormalization,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(epsilon=1e-5, name='norm1')
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, name='attn')

        self.drop_path = DropPath(drop_path) if drop_path > 0. else keras.layers.Identity()
        self.norm2 = norm_layer(epsilon=1e-5, name='norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, name='mlp')

        # if self.shift_size > 0:
        #     self.attn_mask = tf.constant(self.calculate_mask(self.input_resolution))
        # else:
        #     self.attn_mask = None

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size[0], x_size[1]
        img_mask = np.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = np.reshape(mask_windows, [-1, self.window_size * self.window_size])
        attn_mask = mask_windows[:, None] - mask_windows[:, :, None]
        attn_mask = np.where(attn_mask == 0, 0.0, -100.0)
        return attn_mask

    def call(self, x, x_size, training=False):
        H, W = x_size[0], x_size[1]
        input_shape = tf.shape(x)
        b, l, c = input_shape[0], input_shape[1], input_shape[2]
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, [b, H, W, c])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition_layer(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = tf.reshape(x_windows,
                               [-1, self.window_size * self.window_size, c])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size

        # tf.cond()
        # if tf.reduce_all(tf.equal(self.input_resolution, x_size)):
        #     attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # else:
        #     attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size))
        # attn_windows = self.attn(x_windows, mask=self.attn_mask)
        # attn_windows = self.func1(x_windows, x_size)

        # todo: mask 는 input image의 사이즈가 고정되면 고정적으로 사용 할수 있다.
        attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size))

        # merge windows
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, c])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        x = tf.reshape(x, [b, H * W, c])

        # FFN
        x = shortcut + self.drop_path(x, training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)

        return x

    # @tf.function
    # def func1(self, x_windows, x_size):
    #     if tf.reduce_all(tf.equal(self.input_resolution, x_size)):
    #         attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    #     else:
    #         attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size))
    #     return attn_windows

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(keras.layers.Layer):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=keras.layers.LayerNormalization, **kwargs):
        super().__init__(**kwargs)
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm = norm_layer(epsilon=1e-5, )
        self.reduction = keras.layers.Dense(2 * dim, use_bias=False)

    def call(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        input_shape = tf.shape(x)
        b, seq_len, c = input_shape[0], input_shape[1], input_shape[2]
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = tf.reshape(x, [b, h, w, c])

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = tf.concat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = tf.reshape(x, [b, -1, 4 * c])  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(keras.layers.Layer):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__(**kwargs)
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer
        else:
            self.norm = None

    # def compute_output_shape(self, x):
    #     b, h, w, c = keras.ops.shape(x)
    #     return (b, h*w, c)

    def call(self, x):

        input_shape = tf.shape(x)
        b, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        x = tf.reshape(x, [b, h * w, c])
        # x = keras.ops.swapaxes(x, 1, 2)
        if self.norm is not None:
            x = self.norm(x)  # 归一化
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchUnEmbed(keras.layers.Layer):
    r""" Image to Patch Unembedding

    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__(**kwargs)
        img_size = (img_size, img_size)  # 图像的大小，默认为 224*224
        patch_size = (patch_size, patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # patch 的分辨率
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # patch 的个数，num_patches

        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数

    def call(self, x, x_size):
        input_shape = tf.shape(x)  # 输入 x 的结构
        b, c = input_shape[0], input_shape[2]
        # x = keras.ops.swapaxes(x, 1,2)
        x = tf.reshape(x, [b, x_size[0], x_size[1], c])
        return x


class PixelShuffle(keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        self.scale = scale
        super(PixelShuffle, self).__init__(**kwargs)

    def call(self, x):
        input_shape = tf.shape(x)
        b, hh, hw, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        out_channel = c // (self.scale ** 2)
        h = hh * self.scale
        w = hw * self.scale

        x = tf.reshape(x, [b, hh, hw, out_channel, self.scale, self.scale])
        x = keras.layers.Permute((1, 4, 2, 5, 3))(x)

        return tf.reshape(x, [b, h, w, out_channel])


class Upsample(keras.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, **kwargs):
        super(Upsample, self).__init__(**kwargs)
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                self.add(keras.layers.Conv2D(4 * num_feat, kernel_size=3, padding='same'))
                self.add(PixelShuffle(2))
        elif scale == 3:
            self.add(keras.layers.Conv2D(9 * num_feat, kernel_size=3, padding='same'))
            self.add(PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')


# type: DRCT
# upscale: 4
# in_chans: 3
# img_size: 64
# window_size: 16
# compress_ratio: 3
# squeeze_factor: 30
# conv_scale: 0.01
# overlap_ratio: 0.5
# img_range: 1.
# depths: [6, 6, 6, 6, 6, 6]
# embed_dim: 180
# num_heads: [6, 6, 6, 6, 6, 6]
# mlp_ratio: 2
# upsampler: 'pixelshuffle'
# resi_connection: '1conv'

class DRCT(keras.models.Model):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=16,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=keras.layers.LayerNormalization,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 gc=32,
                 **kwargs
                 ):
        super(DRCT, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio
        self.img_size = img_size
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        mixed_policy = keras.mixed_precision.global_policy()
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = tf.constant(rgb_mean, dtype=mixed_policy.compute_dtype)
            self.mean = tf.reshape(self.mean, [1, 1, 1, 3])
        else:
            self.mean = tf.zeros(1, 1, 1, 1, dtype=mixed_policy.compute_dtype)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = keras.layers.Conv2D(embed_dim, kernel_size=3, padding='same', name='conv_first')

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer(epsilon=1e-5, name='norm') if self.patch_norm else None,
        )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = self.add_weight(shape=[1, num_patches, embed_dim],
                                                      initializer=keras.initializers.TruncatedNormal(stddev=0.02))

        self.pos_drop = keras.layers.Dropout(rate=drop_rate)

        # stochastic depth
        dpr = [v for v in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build
        self.body = []
        for i_layer in range(self.num_layers):
            body_layer = RDG(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                             num_heads=num_heads[i_layer], window_size=window_size, depth=0,
                             shift_size=window_size // 2, mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                             norm_layer=norm_layer, gc=gc, img_size=img_size,
                             patch_size=patch_size)  # x : [b, 4096,96] x_size : [64,64]
            self.body.append(body_layer)
        self.body_norm = norm_layer(epsilon=1e-5, name='norm')
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = keras.layers.Conv2D(embed_dim, kernel_size=3, padding='same', name='conv_after_body')
        elif resi_connection == 'identity':
            self.conv_after_body = keras.layers.Identity()

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = keras.Sequential(
                [keras.layers.Conv2D(num_feat, kernel_size=3, padding='same', name='conv_before_upsample'),
                 keras.layers.LeakyReLU(alpha=1e-2)])
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = keras.layers.Conv2D(num_out_ch, kernel_size=3, padding='same', name='conv_last')

        # super().__init__(input_image, output)
        # self.apply(self._init_weights)

    # Dense trunc_normal with zero bias
    # LayerNorm
    # beta_initializer="zeros",
    # gamma_initializer="ones",

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'absolute_pos_embed'}
    #
    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {'relative_position_bias_table'}

    def forward_features(self, x, training=False):
        input_shape = tf.shape(x)
        x_size = (self.img_size, self.img_size)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.body:
            x = layer(x, x_size=x_size, training=training)

        x = self.body_norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size=x_size)

        return x  # [1, 272, 272, 180]

    def call(self, x, training=False):
        # self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = residual = self.conv_first(x)
            x = self.forward_features(x, training=training)
            x = self.conv_after_body(x) + residual
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean

        return x

    # todo 1.dynamic shape 대한 predict
    # todo 2.fixed shape 대한 predict
    # todo 3.window 경계가 별로 깔끔하지 않은데 .. 기존의 pytorch 코드로 돌려봐도 그런 결과인데 어쩔수 없는 한계이려나
