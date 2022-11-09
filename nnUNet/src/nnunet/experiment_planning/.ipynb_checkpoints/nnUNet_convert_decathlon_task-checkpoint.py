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

"""convert decathlon task"""
########### 修改 ##########
import sys
sys.path.append('./')

from batchgenerators.utilities.file_and_folder_operations import subfolders, subfiles, join, os

from src.nnunet.configuration import default_num_threads
from src.nnunet.experiment_planning.utils import split_4d
from src.nnunet.utilities.file_endings import remove_trailing_slash


def crawl_and_remove_hidden_from_decathlon(folder):
    """crawl and remove hidden from decathlon"""
    folder = remove_trailing_slash(folder)
    assert folder.split('/')[-1].startswith("Task"), "This does not seem to be a decathlon folder. Please give me a" \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr," \
                                                     "labelsTr and imagesTs"
    subf = subfolders(folder, join=False)
    assert 'imagesTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                               "folder that starts with TaskXX and has the subfolders imagesTr, " \
                               "labelsTr and imagesTs"
    assert 'imagesTs' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                               "folder that starts with TaskXX and has the subfolders imagesTr, " \
                               "labelsTr and imagesTs"
    assert 'labelsTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                               "folder that starts with TaskXX and has the subfolders imagesTr, " \
                               "labelsTr and imagesTs"
    _ = [os.remove(i) for i in subfiles(folder, prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'imagesTr'), prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'labelsTr'), prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'imagesTs'), prefix=".")]


import glob
import shutil
import json
import SimpleITK as sitk
import numpy as np
def transform(image,newSpacing, resamplemethod=sitk.sitkNearestNeighbor):
    # 设置一个Filter
    resample = sitk.ResampleImageFilter()
    # 初始的体素块尺寸
    originSize = image.GetSize()
    # 初始的体素间距
    originSpacing = image.GetSpacing()
    print('originSpacing', originSpacing)
    newSize = [
        int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
        int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
        int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
    ]
    print('current size:',newSize)

    # 沿着x,y,z,的spacing（3）
    # The sampling grid of the output space is specified with the spacing along each dimension and the origin.
    resample.SetOutputSpacing(newSpacing)
    # 设置original
    resample.SetOutputOrigin(image.GetOrigin())
    # 设置方向
    resample.SetOutputDirection(image.GetDirection())
    resample.SetSize(newSize)
    # 设置插值方式
    resample.SetInterpolator(resamplemethod)
    # 设置transform
    resample.SetTransform(sitk.Euler3DTransform())
    # 默认像素值   resample.SetDefaultPixelValue(image.GetPixelIDValue())
    return resample.Execute(image)

def raw_convert_task_architecture(raw_path, task_path):
    
    train_dirs = sorted(glob.glob(join(raw_path, 'case_*')))[:210] # [:50]
    test_dirs = sorted(glob.glob(join(raw_path, 'case_*')))[210:] # [250:]
    
    os.makedirs(join(task_path, 'imagesTr'), exist_ok=True)
    os.makedirs(join(task_path, 'labelsTr'), exist_ok=True)
    os.makedirs(join(task_path, 'imagesTs'), exist_ok=True)
    
    # 创建json文件
    dataset_json = { 
                    "name": "kits19", 
                    "description": "MindSpore AI",
                    "reference": "the 2019 Kidney Tumor Segmentation Challenge",
                    "tensorImageSize": "3D",
                    "modality":{"0":"CT"},
                    "labels":{"0":"Background","1": "Kidney","2": "Tumor"},
                    "numTraining": 210, #   50 
                    "numTest": 90, # 50
                    "training":[],
                    "test":[]
                    }
    
    # 创建训练集
    for train_dir in train_dirs:
        case_name = os.path.basename(train_dir)
        
        train_img = join(train_dir, 'imaging.nii.gz')

        img_itk = sitk.ReadImage(train_img)
        print('origin size:', img_itk.GetSize())
        new_itk = transform(img_itk, [3.22, 1.62, 1.62], sitk.sitkBSpline) # sitk.sitkLinear
        sitk.WriteImage(new_itk, join(task_path, 'imagesTr', 'kits_' + case_name + '.nii.gz'))
        # shutil.copy(train_img, join(task_path, 'imagesTr', 'kits_' + case_name + '.nii.gz'))
        
        train_label = join(train_dir, 'segmentation.nii.gz')

        label_itk = sitk.ReadImage(train_label)
        print('origin size:', label_itk.GetSize())
        new_itk = transform(label_itk, [3.22, 1.62, 1.62])
        sitk.WriteImage(new_itk, join(task_path, 'labelsTr', 'kits_' + case_name + '.nii.gz'))
        # shutil.copy(train_label, join(task_path, 'labelsTr', 'kits_' + case_name + '.nii.gz'))
        
        dataset_json["training"].append({"image":'./imagesTr/kits_' + case_name + '.nii.gz', "label":'./labelsTr/kits_' + case_name + '.nii.gz'})
        
    # 创建测试集
    for test_dir in test_dirs:
        case_name = os.path.basename(test_dir)
        test_img = join(test_dir, 'imaging.nii.gz')
        shutil.copy(test_img, join(task_path, 'imagesTs', 'kits_' + case_name + '.nii.gz'))
        
        dataset_json["test"].append('./imagesTs/kits_' + case_name + '.nii.gz')
    
    with open(join(task_path, "dataset.json"),'w') as file:
        json.dump(dataset_json,file,indent=4)
        
def main():
    """The MSD provides data as 4D Niftis with the modality being the first"""

    import argparse
    parser = argparse.ArgumentParser(description="The MSD provides data as 4D Niftis with the modality being the first"
                                                 " dimension. We think this may be cumbersome for some users and "
                                                 "therefore expect 3D niftixs instead, with one file per modality. "
                                                 "This utility will convert 4D MSD data into the format nnU-Net "
                                                 "expects")
    parser.add_argument("-r", help="original raw data path", required=True)
    parser.add_argument("-i", help="Input folder. Must point to a TaskXX_TASKNAME folder as downloaded from the MSD "
                                   "website", required=True)
    parser.add_argument("-p", required=False, default=default_num_threads, type=int,
                        help="Use this to specify how many processes are used to run the script. "
                             "Default is %d" % default_num_threads)
    parser.add_argument("-output_task_id", required=False, default=None, type=int,
                        help="If specified, this will overwrite the task id in the output folder. If unspecified, the "
                             "task id of the input folder will be used.")
    args = parser.parse_args()
    
    # 先将原始下载的kits19的数据组织结构调整成nnUNet的组织结构
    print('-----begin convert kist9 dirs architecture')
    raw_convert_task_architecture(args.r, args.i)

    crawl_and_remove_hidden_from_decathlon(args.i)

    split_4d(args.i, args.p, args.output_task_id)
    print('-----finish convert kits19 dirs architecture')


if __name__ == "__main__":
    main()
