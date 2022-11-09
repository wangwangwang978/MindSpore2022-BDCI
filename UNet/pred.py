# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import logging
from mindspore import context, Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.data_loader import create_dataset, create_pred_dataset
from src.unet_medical import UNetMedical
from src.unet_nested import NestedUNet, UNet
from src.utils import UnetEval, TempLoss, dice_coeff
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

import mindspore.ops.operations as F
from mindspore.common.tensor import Tensor
import glob
import os
import SimpleITK as sitk
from src.data_loader import scaleIntensityRange
import numpy as np
import mindspore as ms
from mindspore import nn

@moxing_wrapper()
def test_net(data_dir,
             ckpt_path,
             cross_valid_ind=1):
    if config.model_name == 'unet_medical':
        net = UNetMedical(n_channels=config.num_channels, n_classes=config.num_classes)
    elif config.model_name == 'unet_nested':
        net = NestedUNet(in_channel=config.num_channels, n_class=config.num_classes, use_deconv=config.use_deconv,
                         use_bn=config.use_bn, use_ds=False)
    elif config.model_name == 'unet_simple':
        net = UNet(in_channel=config.num_channels, n_class=config.num_classes)
    else:
        raise ValueError("Unsupported model: {}".format(config.model_name))
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    net = UnetEval(net, eval_activate=config.eval_activate.lower())
    if hasattr(config, "dataset") and config.dataset != "ISBI":
        split = config.split if hasattr(config, "split") else 0.8
        pred_dataset = create_pred_dataset(data_dir, config.image_size, 1, 1,
                                                   num_classes=config.num_classes, is_train=False,
                                                   eval_resize=config.eval_resize, split=split, shuffle=False)
    else:
        _, pred_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False,
                                          do_crop=config.crop, img_size=config.image_size)
    # model = Model(net, loss_fn=TempLoss(), metrics={"dice_coeff": dice_coeff(show_eval=config.show_eval)})
    
    print("============== Starting Prediction ============")
    softmax = nn.Softmax(1)
    
    pred_folders = sorted(glob.glob(os.path.join(config.data_path, 'case_*')))[210:]
    for index in range(len(pred_folders)):
        img_path = os.path.join(pred_folders[index], 'imaging.nii.gz')
        img = sitk.ReadImage(img_path)
        spacing = img.GetSpacing()
        img_arr = sitk.GetArrayFromImage(img)
        H, W, D = img_arr.shape
        print('processing img_path {} img_arr shape{} '.format(img_path, img_arr.shape))
        prediction = np.zeros(img_arr.shape, np.uint8)
        for d in range(D):
            img_2d = img_arr[:,:,d]
            img_2d = scaleIntensityRange(img_2d)
            img_2d = img_2d[None,None,...].astype(np.float32) # 扩充一个维度 不支持float64
            img_2d = Tensor(img_2d)
            out = net(img_2d)
            out = out.transpose(0, 3, 1, 2) # NHWC -> NCHW 
            # print('out shape: ', out.shape)
            pred = softmax(out).argmax(1)
            pred = pred.squeeze(0).asnumpy() 
            # print('pred shape: ', pred.shape)
            prediction[:,:,d] = pred

        prd_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
        prd_itk.SetSpacing(spacing)
        pred_save_dir= os.path.join(config.pred_save_path, os.path.basename(pred_folders[index]))
        os.makedirs(pred_save_dir, exist_ok=True)
        sitk.WriteImage(prd_itk, os.path.join(pred_save_dir, "segmentation.nii.gz"))
        print(os.path.join(pred_save_dir, "segmentation.nii.gz"), ' saved!')
    print("============== Ending Prediction ============")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    if config.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
    test_net(data_dir=config.data_path,
             ckpt_path=config.checkpoint_file_path,
             cross_valid_ind=config.cross_valid_ind)
