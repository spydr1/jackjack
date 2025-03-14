"""
MIT license
https://github.com/ming053l/DRCT/blob/main/drct/archs/DRCT_arch.py
"""
import math
import tqdm
import tensorflow as tf
import tensorflow.keras as keras

from jackjack.super_resolution.legacy.real_esrgan.image_utils import *

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

    # def compute_output_shape(self, x_shape):
    #     return x_shape

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
        self.num_feat = num_feat
        # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
        self.attention = keras.Sequential(
            [
                keras.layers.GlobalAvgPool2D(),
                keras.layers.Conv2D(self.num_feat // squeeze_factor, kernel_size=1, padding='valid'),
                keras.layers.Activation(keras.activations.relu),  # inplace=True
                keras.layers.Conv2D(self.num_feat, kernel_size=1, padding='valid'),
                keras.layers.Activation(keras.activations.sigmoid),
            ]
        )

    # def compute_output_shape(self, x_shape):
    #     batch_size = x_shape[0]
    #     return tf.TensorShape([batch_size, None, None, self.num_feat])

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
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.fc1 = keras.layers.Dense(self.hidden_features, name='fc1')
        self.act = keras.layers.Activation(act_layer)
        self.fc2 = keras.layers.Dense(self.out_features, name='fc2')
        self.drop = keras.layers.Dropout(drop)

    # def compute_output_shape(self, x_shape):
    #     batch_size = x_shape[0]
    #     return tf.TensorShape([batch_size, None, self.out_features])

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    shape = tf.shape(x)
    b, h, w, c = shape[0], shape[1], shape[2], shape[3]
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
        self.qkv_bias = qkv_bias
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

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


        self.qkv = keras.layers.Dense(self.dim * 3, use_bias=self.qkv_bias, name='qkv')
        self.attn_drop = keras.layers.Dropout(self.attn_drop)
        self.proj = keras.layers.Dense(self.dim, name='proj')

        self.proj_drop = keras.layers.Dropout(self.proj_drop)

        self.softmax = keras.layers.Softmax(axis=-1)

    # todo :
    def build(self, input_shape=None):
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="relative_position_bias_table"
        )  # 2*Wh-1 * 2*Ww-1, nH

    # def compute_output_shape(self, input_shape):
    #     return input_shape

    def call(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        Returns:
             x: (num_windows*B, N, C)
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
        x = tf.reshape(x, [b, n, c]) # [num_windows*B, N, C]
        x = self.proj(x)  # z = self.proj(x)
        x = self.proj_drop(x)
        return x


# wa = WindowAttention(dim=256,window_size=(8,8), num_heads=4)
# wa.compute_output_shape([None,None,256])
# wa.build([None,None,256])
# wa(tf.random.normal([32*32,64,256]))
# wa
# inputs = keras.Input([32*32,64,256])
# model = keras.Model(inputs=inputs, outputs=wa(inputs))
# model(tf.random.normal([32*32,64,256]))

class RDG(keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, qk_scale,
                 drop, attn_drop, drop_path, norm_layer, gc, patch_size, img_size, **kwargs):
        super(RDG, self).__init__(**kwargs)
        self.dim = dim
        self.gc = gc

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
        self.pe = PatchEmbed()

        self.pue = PatchUnEmbed(embed_dim=dim)
        ### no weight.

    # def compute_output_shape(self, input_shape):
    #     batch_size = input_shape[0]
    #     return tf.TensorShape([batch_size, None, self.dim])

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
        self.norm_layer = norm_layer
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.act_layer = act_layer
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # if self.shift_size > 0:
        #     self.attn_mask = tf.constant(self.calculate_mask(self.input_resolution))
        # else:
        #     self.attn_mask = None

        self.norm1 = self.norm_layer(epsilon=1e-5, name='norm1')
        self.attn = WindowAttention(
            self.dim, window_size=(self.window_size, self.window_size), num_heads=self.num_heads,
            qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, attn_drop=self.attn_drop, proj_drop=self.drop, name='attn')

        self.drop_path = DropPath(self.drop_path) if self.drop_path > 0. else keras.layers.Identity()
        self.norm2 = self.norm_layer(epsilon=1e-5, name='norm2')
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=self.act_layer, drop=self.drop, name='mlp')

    # def build(self, input_shape):
    #     self.norm1 = self.norm_layer(epsilon=1e-5, name='norm1')
    #     self.attn = WindowAttention(
    #         self.dim, window_size=(self.window_size, self.window_size), num_heads=self.num_heads,
    #         qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, attn_drop=self.attn_drop, proj_drop=self.drop, name='attn')
    #
    #     self.drop_path = DropPath(self.drop_path) if self.drop_path > 0. else keras.layers.Identity()
    #     self.norm2 = self.norm_layer(epsilon=1e-5, name='norm2')
    #     mlp_hidden_dim = int(self.dim * self.mlp_ratio)
    #     self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=self.act_layer, drop=self.drop, name='mlp')

    # @tf.function(jit_compile=False)
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size[0], x_size[1]
        img_mask = tf.zeros((H, W), dtype=tf.int32)  # 1 H W 1

        # h_slices = (slice(0, -self.window_size),
        #             slice(-self.window_size, -self.shift_size),
        #             slice(-self.shift_size, None))
        # w_slices = (slice(0, -self.window_size),
        #             slice(-self.window_size, -self.shift_size),
        #             slice(-self.shift_size, None))
        # cnt = 0
        # for h in h_slices:
        #     for w in w_slices:
        #         img_mask[:, h, w, :] = cnt
        #         cnt += 1



        cnt = tf.range(9)

        h_slices = tf.convert_to_tensor([[tf.constant(0), tf.subtract(H,self.window_size)],
                                [H - self.window_size, H -self.shift_size],
                                [H -self.shift_size, H]], dtype=tf.int32)
        w_slices = tf.convert_to_tensor([[tf.constant(0), W - self.window_size],
                                [W - self.window_size, W -self.shift_size],
                                [W -self.shift_size, W]], dtype=tf.int32)

        for i in range(3):
            for j in range(3):
                row_start = h_slices[i,0]
                row_end = h_slices[i,1]
                col_start = w_slices[j,0]
                col_end = w_slices[j,1]
                row_indices, col_indices = tf.meshgrid(tf.range(row_start, row_end), tf.range(col_start, col_end),indexing='ij')
                indices = tf.stack([tf.reshape(row_indices, [-1]), tf.reshape(col_indices, [-1])], axis=1)

                indices_shape = tf.shape(indices)
                img_mask = tf.cond(tf.greater(indices_shape[0], 0), lambda : tf.tensor_scatter_nd_update(img_mask, indices, tf.repeat(cnt[3 * i + j], indices_shape[0])),lambda: img_mask )
                # img_mask = tf.cond(tf.reduce_prod(indices_shape)> 0, lambda : tf.tensor_scatter_nd_update(img_mask, indices, tf.repeat(cnt[3*i+j], indices_shape[0])), lambda: img_mask)

        img_mask = img_mask[None,:,:,None]
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
        attn_mask = mask_windows[:, None] - mask_windows[:, :, None]
        attn_mask = tf.where(attn_mask == 0, 0.0, -100.0)
        return attn_mask


    # def compute_output_shape(self, input_shape):
    #     b, c = input_shape
    #     return tf.TensorShape([b, None, c])

    def call(self, x, x_size, training=False):

        H, W = x_size[0], x_size[1]

        input_shape = tf.shape(x)
        b, l, c = input_shape[0], input_shape[1], input_shape[2] # [b, height * width , hidden dim]
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
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
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
        mask = tf.cast(self.calculate_mask(x_size), tf.float32)
        attn_windows = self.attn(x_windows, mask=mask) # num_windows*B, N, C

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

    # def compute_output_shape(self, x_shape):
    #     b, c  = x_shape[0], x_shape[2]
    #     return tf.TensorShape([b, None, c*4])

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
        x = tf.reshape(x, [b, -1, c*4])  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x


# https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/models/swin_mlp.py#L300
# super-resolution task라서 원래의 코드와 비교해 patch siz = 1 인 특수한 상황이라고 볼수 있을 것 같다.
# 사실상 PatchEmbedding이라고 볼수 없음.
class PatchEmbed(keras.layers.Layer):
    r""" Image to Patch Embedding

    Args:
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, norm_layer=None, **kwargs):
        super().__init__(**kwargs)
        if norm_layer is not None:
            self.norm = norm_layer
        else:
            self.norm = None

    # def compute_output_shape(self, x_shape):
    #     b, c  = x_shape[0], x_shape[3]
    #     return tf.TensorShape([b, None, c])

    def call(self, x):
        input_shape = tf.shape(x)
        b, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        x = tf.reshape(x, [b, h * w, c])
        # x = keras.ops.swapaxes(x, 1, 2)
        if self.norm is not None:
            x = self.norm(x)  # 归一化
        return x

# embed = PatchEmbed()
# embed.compute_output_shape([4, None,None,3])
# inputs = keras.Input([None,None,3])
# model = keras.Model(inputs=inputs, outputs=embed(inputs))
# model(tf.random.normal([4,64,64,3]))

class PatchUnEmbed(keras.layers.Layer):
    r""" Image to Patch Unembedding

    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(self, embed_dim,  **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    # def compute_output_shape(self, x_shape):
    #     batch_size  = x_shape[0]
    #     return tf.TensorShape([batch_size, None, None, self.embed_dim])

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

    # def compute_output_shape(self, input_shape):
    #     b, h, w, c  = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    #     return tf.TensorShape([b, None,None, c // (self.scale**2)])

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
                 input_specs: keras.layers.InputSpec = keras.layers.InputSpec(shape=[None, None, None, 3]),
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

        # ------------------------- 0, input ------------------------- #

        x = inputs = keras.Input(shape=input_specs.shape[1:])
        x = (x - self.mean) * self.img_range

        if tf._tf_uses_legacy_keras:
            input_shape = tf.shape(inputs) #.shape
            h, w = inputs.shape[1:3]
            if h and w :
                assert h == w, "height and width must be same. (square)"
        else :
            input_shape = inputs.shape# .shape
            # todo : keras 3 에서는 tf.shape 를 사용할수 없네 ..
            h, w = input_shape[1], input_shape[2]
            if None in input_shape[1:]:
                raise ValueError("You are using keras3. Input shape must be specific. Please set input_specs = keras.layers.InputSpec(shape=[b, h, w, 3])")
            else:
                assert h == w, "height and width must be same. (square)"

        # ------------------------- 1, shallow feature extraction ------------------------- #

        x = residual = keras.layers.Conv2D(embed_dim, kernel_size=3, padding='same', name='conv_first')(x)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        x = PatchEmbed(
            norm_layer=norm_layer(epsilon=1e-5, name='norm') if self.patch_norm else None,
        )(x)

        # merge non-overlapping patches into image

        # absolute position embedding
        # warning : super-resolution task 에서는 아예 쓰지 않는다.
        # if self.ape:
        #     # assert (input_specs.shape[1] and input_specs.shape[2]) is not None, \
        #     #     "Input_specs must be [None, height, width, 3]. " \
        #     #     "Please set specific height, width."
        #
        #     img_size = (img_size, img_size)
        #     patch_size = (patch_size, patch_size)
        #     patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        #     self.img_size = img_size
        #     self.patch_size = patch_size
        #     self.patches_resolution = patches_resolution
        #     self.num_patches = patches_resolution[0] * patches_resolution[1]
        #     self.absolute_pos_embed = self.add_weight(shape=[1, self.num_patches, embed_dim],
        #                                               initializer=keras.initializers.TruncatedNormal(stddev=0.02))

        x = keras.layers.Dropout(rate=drop_rate)(x)

        # stochastic depth
        dpr = [v for v in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        x_size = (input_shape[1], input_shape[2])
        for i_layer in range(self.num_layers):
            x = RDG(dim=embed_dim, input_resolution=(img_size, img_size),
                             num_heads=num_heads[i_layer], window_size=window_size, depth=0,
                             shift_size=window_size // 2, mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                             norm_layer=norm_layer, gc=gc, img_size=img_size,
                             patch_size=patch_size)(x=x, x_size=x_size)  # x : [b, 4096,96] x_size : [64,64]
        x = keras.layers.LayerNormalization(epsilon=1e-5, name='body_norm')(x)
        x = PatchUnEmbed(embed_dim=embed_dim)(x=x, x_size=x_size)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            x = keras.layers.Conv2D(embed_dim, kernel_size=3, padding='same', name='conv_after_body')(x) + residual
        elif resi_connection == 'identity':
            x = keras.layers.Identity()(x)

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = keras.Sequential(
                [keras.layers.Conv2D(num_feat, kernel_size=3, padding='same', name='conv_before_upsample'),
                 keras.layers.LeakyReLU(alpha=1e-2)])(x)
            x = Upsample(upscale, num_feat)(x)
            x = keras.layers.Conv2D(num_out_ch, kernel_size=3, padding='same', name='conv_last')(x)
            outputs = x / self.img_range + self.mean

        logging.info("*************************\n"
                     "Restriction of algorithm\n"
                     "Resolution must be\n"
                     "1. multiple of window size.\n"
                     "2. greater than 64.\n"
                     "*************************")

        super().__init__(inputs=inputs, outputs=outputs)

    # def compute_output_shape(self, input_shape):
    #     return tf.TensorShape([input_shape[0], input_shape[1]*self.upscale, input_shape[2]*self.upscale, input_shape[3]])

    def call(self, inputs, **kwargs):
        b, h, w, c = inputs.shape
        assert h % self.window_size ==0  and w % self.window_size==0, f"Input resolution must be multiple of window size{self.window_size}."
        assert (h and w) >= 64, f"Input resolution must be greater than 64."

        return super().call(inputs, **kwargs)

    def predict_dynamic_shape(self,
                              low_resolution_image):
        """
        :param low_resolution_image: numpy array of single Image. not batched.
            1. shape : [h,w,3]
            2. range : 0~255
            3. rgb channel
        :return:
        """

        logging.warning("You are using dynamic shape prediction. It is not recommended. \n"
                        "1. problem of OOM, low speed. \n"
                        "2. fixed batch size to 1. \n"
                        "3. impossible to parallel computational")

        # ***todo : 이부분 설명
        window_size = self.window_size
        h_old, w_old, c = low_resolution_image.shape
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old

        new_shape = [h_old + h_pad, w_old + w_pad, 3]
        padded_image = np.zeros(new_shape)
        padded_image[:h_old, :w_old] = low_resolution_image
        # ******************

        x = padded_image/ 255
        np_sr_image = self(x).numpy()
        np_sr_image = np.clip(np_sr_image, 0, 1)
        sr_img = (np_sr_image * 255).astype(np.uint8)

        return sr_img[0]

    def predict(
            self,
            low_resolution_image,
            batch_size=1,
            padding = 0,
    ):
        """
        :param low_resolution_image: numpy array of Single Image. It is not batched.
            1. shape : [h,w,3]
            2. range : 0~255
            3. rgb channel
        :param batch_size:
        :return:
        """
        h = self.input_spec[0].shape[1]
        patches, p_shape = split_image_into_overlapping_patches(low_resolution_image, patch_size=h, overlap_size=padding)
        patches = patches / 255
        num_of_patch = patches.shape[0]
        result = []
        for i in tqdm.trange(0, num_of_patch, batch_size):
            result += [self.predict_on_batch(patches[i:i + batch_size])]
            # todo : self(patches[i:i + batch_size])

        result_concat = np.concatenate(result, axis=0)
        sr_image = result_concat.clip(0, 1)

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], self.upscale)) + (3,)
        scaled_image_shape = tuple(np.multiply(low_resolution_image.shape[0:2], self.upscale)) + (3,)
        np_sr_image = stich_together(
            sr_image, padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape, padding_size=padding * self.upscale
        )
        np_sr_image = np.clip(np_sr_image, 0, 1)
        sr_img = (np_sr_image * 255).astype(np.uint8)

        return sr_img