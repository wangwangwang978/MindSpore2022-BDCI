#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
########### 修改 ##########
import sys
sys.path.append('./')
from skimage.transform import resize #########

from src.nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from mindspore import nn

class nnUNetTrainerV2_SGD(nnUNetTrainerV2):
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.lr = []
        for i in range(0, self.max_num_epochs):
            self.lr.append(self.poly_lr(epoch=i))
        # self.optimizer = nn.SGD(self.network.trainable_params(), self.initial_lr, weight_decay=self.weight_decay,
        #                                  momentum=0.99)
        self.optimizer = nn.Momentum(params=self.network.trainable_params(), learning_rate=0.01, momentum=0.9)
        self.lr_scheduler = None

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs

        """
        return super().on_epoch_end()

if __name__ == '__main__':
    from src.nnunet.network_architecture.generic_UNet import Generic_UNet
    net = Generic_UNet(input_channels=1, base_num_features=2, num_classes=3, num_pool=None)
    optimizer = nn.Momentum(params=net.trainable_params(), learning_rate=0.01, momentum=0.9)

    optimizer.param_groups[0]["momentum"] = 0.95