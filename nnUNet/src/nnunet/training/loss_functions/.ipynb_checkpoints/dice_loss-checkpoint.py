# Copyright 2022 Huawei Technologies Co., Ltd
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

"""dice loss module"""
########### 修改 ##########
import sys
sys.path.append('./')

import mindspore.ops as ops
from mindspore import nn, Tensor
from mindspore import dtype as mstype
import numpy as np
from mindspore.nn.loss.loss import LossBase
import mindspore as ms
import mindspore.ops as ops
from mindspore._checkparam import Validator as validator

from src.nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss, RobustCrossEntropyLoss2d
from src.nnunet.utilities.nd_softmax import softmax_helper


class SoftDiceLoss(LossBase):
    """soft dice loss module"""

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1e-5, loss_type='3d'):
        super(SoftDiceLoss, self).__init__()
        self.mean = ops.ReduceMean()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.reshape = ops.Reshape()
        self.zeros = ops.Zeros()
        self.dc = nn.DiceLoss(smooth=smooth)
        
    def construct(self, net_output, target):            
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
        
        return self.dc(net_output, target)
    
class DC_and_CE_loss(LossBase):
    """Dice and cross entrophy loss"""
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1.0, weight_dice=0, # 0，1.0
                 log_dice=False, ignore_label=None):

        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate

        if soft_dice_kwargs["loss_type"] == '3d':
            self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        else:
            self.ce = RobustCrossEntropyLoss2d(**ce_kwargs)

        self.transpose = ops.Transpose()
        self.ignore_label = ignore_label
        self.reshape = ops.Reshape()

        # self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.dc = nn.DiceLoss()
        
        # print('soft_dice_kwargs, ce_kwargs ', soft_dice_kwargs, ce_kwargs)

    def construct(self, net_output, target):
        """construct network"""
        # print('DC_and_CE_loss Loss net_output.shape, target.shape', net_output.shape, target.shape)
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss # 相加还是相减
        return result

def softmax_helper(data):
    """mindspore softmax"""
    return ops.Softmax(1)(data)


import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor



class DiceLoss(LossBase):
    
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = validator.check_positive_float(smooth, "smooth")
        self.reshape = ops.Reshape()

    def construct(self, logits, label):
        # 进行维度校验，维度必须相等。（输入必须是tensor）
        # _check_shape(logits.shape, label.shape)
        # 求交集，和dice系数一样的方式
        intersection = self.reduce_sum(self.mul(logits.view(-1), label.view(-1)))
        # 求并集，和dice系数一样的方式
        unionset = self.reduce_sum(self.mul(logits.view(-1), logits.view(-1))) + \
                   self.reduce_sum(self.mul(label.view(-1), label.view(-1)))
        
        # 利用公式进行计算
        single_dice_coeff = (2 * intersection) / (unionset + self.smooth)
        dice_loss = 1 - single_dice_coeff / label.shape[0]

        return dice_loss

class MultiClassDiceLoss(LossBase):
    
    def __init__(self, weights=None, ignore_indiex=None, activation=ops.Softmax(axis=1)):
        super(MultiClassDiceLoss, self).__init__()
        
        # 利用Dice系数
        self.binarydiceloss = DiceLoss(smooth=1e-5)
        # 权重是一个Tensor，应该和分类数的维度一样：Tensor of shape `[num_classes, dim]`。
        self.weights = weights if weights is None else validator.check_value_type("weights", weights, [Tensor])
        # 要忽略的类别序号
        self.ignore_indiex = ignore_indiex if ignore_indiex is None else \
            validator.check_value_type("ignore_indiex", ignore_indiex, [int])
        # 使用激活函数
        self.activation = nn.get_activation(activation) if isinstance(activation, str) else activation
        if activation is not None and not isinstance(self.activation, nn.Cell):
            raise TypeError("The activation must be str or Cell, but got {}.".format(activation))
        self.activation_flag = self.activation is not None
        self.reshape = ops.Reshape()

    def construct(self, logits, label):
        # 先定义一个loss，初始值为0
        total_loss = 0
      
        # 如果使用激活函数
        if self.activation_flag:
            logits = self.activation(logits)
        # 按照标签的维度的第一个数进行遍历
        for i in range(label.shape[1]):
            if i != self.ignore_indiex:
                dice_loss = self.binarydiceloss(logits[:, i], label[:, i])
                if self.weights is not None:
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/label.shape[1]
    
if __name__ == '__main__':
    
    loss = MultiClassDiceLoss(weights=None, ignore_indiex=None, activation="softmax")
    y_pred = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mstype.float32)
    y = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mstype.float32)
    output = loss(y_pred, y)
    print(output)
    
    loss = SoftDiceLoss(apply_nonlin=softmax_helper, **{'batch_dice': True, 'smooth': 1e-05, 'do_bg': False})
    logits = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mstype.float32)
    labels = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mstype.float32)
    output = loss(logits, labels)
    print(output)
    
    