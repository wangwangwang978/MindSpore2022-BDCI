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
"""
Preprocess dataset.
Images within one folder is an image, the image file named `"image.png"`, the mask file named `"mask.png"`.
"""
import os
import cv2
import numpy as np
from src.model_utils.config import config
from scipy import ndimage

import multiprocessing
from multiprocessing import Pool
import SimpleITK as sitk
from PIL import Image
import glob

np.random.seed(1)
def savePalette(image_array, save_path):
    mask = image_array.convert("L")
    palette=[]
    for j in range(256):
        palette.extend((j,j,j))    
        palette[:3*10]=np.array([
                                [0, 0, 0], # 黑色非组织区域 label 0
                                [0,255,0], # 绿色 label 1 
                                [0,0,255], # 蓝色：label 2
                                [255,255,0], # 黄色 label 3 
                                [255,0,0], # 红色：label 4
                                [0,255,255],# 淡蓝色：label 5
                            ], dtype='uint8').flatten()
    mask = mask.convert('P')
    mask.putpalette(palette)
    if not os.path.exists(os.path.dirname(save_path)):#检查目录是否存在
        os.makedirs(os.path.dirname(save_path))#如果不存在则创建目录
    mask.save(save_path)

def multi_processing():
    processes = multiprocessing.cpu_count()
    with Pool(processes=processes) as pool:
        results = [pool.apply_async(staple,
                                    args=(maskfile, args.inputdirs,
                                          args.outputdir, args.undecidedlabel))
                   for maskfile in maskfiles]
        _ = [_.get() for _ in results]
    print("Done")

def preprocess(case_dir,save_dir):
    img_path = os.path.join(case_dir, 'imaging.nii.gz')
    seg_path = os.path.join(case_dir, 'segmentation.nii.gz')

    if not os.path.exists(img_path) and not os.path.exists(seg_path):
        print('img_path {} image has processed!'.format(img_path))
        return 

    img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img)

    print('img_path {} image shape {} type {}'.format(img_path, img_arr.shape, img_arr.dtype))
    seg = sitk.ReadImage(seg_path)
    seg_arr = sitk.GetArrayFromImage(seg)
    print('seg_path {} seg shape {} type {}'.format(seg_path, seg_arr.shape, seg_arr.dtype))
    img_arr = ndimage.zoom(img_arr, (1.0, 1.0, 0.5), order=3)
    seg_arr = ndimage.zoom(seg_arr, (1.0, 1.0, 0.5), order=0)
    H, W, D = img_arr.shape
    print('finale down_scale shape {}'.format(img_arr.shape))
    for d in range(D):
        if np.max(seg_arr[:,:,d]) > 0: # 含有label类
            type_slice = 'foreground'
            np.savez(os.path.join(save_dir, type_slice, os.path.basename(case_dir) + '_' + '%04d.npz'%d), image=img_arr[:,:,d], mask=seg_arr[:,:,d])
        elif np.random.rand() < 0.187:
            type_slice = 'background'
            np.savez(os.path.join(save_dir, type_slice, os.path.basename(case_dir) + '_' + '%04d.npz'%d), image=img_arr[:,:,d], mask=seg_arr[:,:,d])
    # 删除文件，以防止空间不够 （但不删除文件夹）       
    os.remove(img_path)
    os.remove(seg_path)
    print('img_path removed {}'.format(img_path))
    print('seg_path removed {}'.format(seg_path))
    
def preprocess_kits19(data_dir, save_dir):

    print("========== start preprocess kits19 dataset ==========")
    os.makedirs(os.path.join(save_dir, 'foreground'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'background'), exist_ok=True)
    dirs = sorted(glob.glob(os.path.join(data_dir, 'case_*')))[:210]

    processes = 16 # multiprocessing.cpu_count() # 根据CPU能力自行设定
    with Pool(processes=processes) as pool:
        results = [pool.apply_async(preprocess,
                                    args=(case_dir,save_dir))
                   for case_dir in dirs]
        _ = [_.get() for _ in results]
    print("========== end preprocess dataset ==========")
        
if __name__ == '__main__':
    preprocess_kits19(config.raw_data_path, config.data_path)
