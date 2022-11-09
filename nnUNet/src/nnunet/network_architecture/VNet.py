# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Vnet network"""
import mindspore.nn as nn
import mindspore.ops
from copy import deepcopy
import mindspore as ms
from mindspore.common.initializer import initializer, HeNormal
import numpy as np

from src.nnunet.network_architecture.initialization import InitWeights_He
from src.nnunet.network_architecture.neural_network import SegmentationNetwork
from src.nnunet.utilities.nd_softmax import softmax_helper


def ELUCons(elu, nchannels):
    """activation function"""

    if elu:
        return mindspore.ops.Elu()
    return nn.PReLU(nchannels)


class LUConv(nn.Cell):
    """convolution with activation function and BN"""

    def __init__(self, nchannels, elu):
        super(LUConv, self).__init__()
        self.relu = ELUCons(elu, nchannels)
        self.conv = nn.Conv3d(nchannels, nchannels, kernel_size=(5, 5, 5), pad_mode='pad', padding=2)
        self.bn = nn.BatchNorm3d(nchannels)

    def construct(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


def _make_nConv(nchannels, depth, elu):
    """make convolution layers"""

    layers = []
    if depth == 1:
        return LUConv(nchannels, elu)
    for _ in range(depth):
        layers.append(LUConv(nchannels, elu))
    return nn.SequentialCell(*layers)


class InputTransition(nn.Cell):
    """input transition module"""

    def __init__(self, out_channels, elu):
        super(InputTransition, self).__init__()
        self.conv = nn.Conv3d(1, 16, kernel_size=(5, 5, 5), pad_mode='pad', padding=2)
        self.bn = nn.BatchNorm3d(16)
        self.relu = ELUCons(elu, 16)
        self.cat = mindspore.ops.Concat(axis=1)

    def construct(self, x):
        out = self.bn(self.conv(x))
        x16 = self.cat((x, x, x, x, x, x, x, x,
                        x, x, x, x, x, x, x, x))
        out = self.relu(mindspore.ops.tensor_add(out, x16))
        return out


class DownTransition(nn.Cell):
    """down transition module"""

    def __init__(self, in_channels, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        out_channels = 2 * in_channels
        self.dropout = dropout
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(2, 2, 2), stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu1 = ELUCons(elu, out_channels)
        self.relu2 = ELUCons(elu, out_channels)
        self.dropout_op = mindspore.ops.Dropout3D()
        self.ops = _make_nConv(out_channels, nConvs, elu)

    def construct(self, x):
        down = self.relu1(self.bn(self.down_conv(x)))
        if self.dropout:
            out, _ = self.dropout_op(down)
        else:
            out = down
        out = self.ops(out)
        out = self.relu2(mindspore.ops.tensor_add(out, down))
        return out


class UpTransition(nn.Cell):
    """up transition module"""

    def __init__(self, in_channels, out_channels, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.dropout = dropout
        self.up_conv = nn.Conv3dTranspose(in_channels, out_channels // 2, kernel_size=(2, 2, 2), stride=2)
        self.bn = nn.BatchNorm3d(out_channels // 2)
        self.relu1 = ELUCons(elu, out_channels // 2)
        self.relu2 = ELUCons(elu, out_channels)
        self.dropout_op1 = mindspore.ops.Dropout3D()
        self.dropout_op2 = mindspore.ops.Dropout3D()
        self.ops = _make_nConv(out_channels, nConvs, elu)
        self.cat = mindspore.ops.Concat(axis=1)

    def construct(self, x, skipx):
        """up transition module construct"""

        if self.dropout:
            out, _ = self.dropout_op1(x)
            skipx_dropout, _ = self.dropout_op2(skipx)
        else:
            out = x
            skipx_dropout = skipx
        out = self.relu1(self.bn(self.up_conv(out)))
        xcat = self.cat((out, skipx_dropout))
        out = self.ops(xcat)
        out = self.relu2(mindspore.ops.tensor_add(out, xcat))
        return out


class OutputTransition(nn.Cell):
    """output transition module"""

    def __init__(self, in_channels, elu):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 2, kernel_size=(5, 5, 5), pad_mode='pad', padding=2)
        self.bn = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=(3, 3, 3), pad_mode='pad', padding=1)
        self.relu = ELUCons(elu, 2)
        self.softmax = nn.Softmax(axis=1)
        self.transpose = mindspore.ops.Transpose()

    def construct(self, x):
        out = self.relu(self.bn(self.conv1(x)))
        out = self.conv2(out)
        B, C, X, Y, Z = out.shape
        out = out.view(B, C, -1)
        out = self.softmax(out)
        out = out[:, 0, :].view(B, X, Y, Z)
        return out


class VNet(nn.Cell):
    """vnet model"""

    def __init__(self, dropout=True, elu=True):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=dropout)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=dropout)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=dropout)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=dropout)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu)

    def construct(self, x):
        """vnet construct"""

        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)

        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

def print_module_training_status(module):
    """print module training status"""
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Dropout3d, nn.Dropout2d,
                           nn.Dropout, nn.InstanceNorm3d, nn.InstanceNorm2d,
                           nn.InstanceNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
        print(str(module), module.training)

class generic_VNet(SegmentationNetwork):
    """Generic UNet Parameters"""
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (96, 160, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=True, convolutional_upsampling=False, # False, True, True
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        """
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        print('upscale_logits', upscale_logits)
        if nonlin_kwargs is None:
            nonlin_kwargs = {'alpha': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'has_bias': True, "pad_mode": "pad"}
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        if self.weightInitializer is not None:
            if isinstance(self.weightInitializer, InitWeights_He):
                self.init_weights(self.cells())

    def init_weights(self, cell):
        """init_weights"""
        for op_cell in cell:

            if isinstance(op_cell, nn.SequentialCell):
                self.init_weights(op_cell)
            else:
                if isinstance(op_cell, (nn.Conv3d, nn.Conv2d, nn.Conv2dTranspose, nn.Conv3dTranspose)):
                    op_cell.weight.set_data = initializer(HeNormal(), op_cell.weight.shape, ms.float32)
                    if op_cell.bias is not None:
                        op_cell.bias.set_data = initializer(0, op_cell.bias.shape, ms.float32)

    def construct(self, x):
        """construct network"""
        # print('-------Generic_UNet input x---------',x.shape)


        
        if self._deep_supervision and self.do_ds:
            return ([seg_outputs[-1]] + [seg_outputs[3], seg_outputs[2], seg_outputs[1], seg_outputs[0]])
        if not self._deep_supervision and not self.do_ds:
            return seg_outputs[-1]
        return None

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (
                npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
        return tmp


if __name__ == '__main__':
    from mindspore import Tensor
    import numpy as np
    inputs = Tensor(np.array(np.random.randn()))
    net = VNet()
    
