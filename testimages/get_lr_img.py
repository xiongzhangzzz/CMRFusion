import torch
from PIL import Image
import sys
import os
import numpy as np
hr_img_path1 = "./ir/"
hr_img_path2 = "./vi/"
lr_img_path1 =  "./ir_0.25/"
lr_img_path2 =  "./ir_0.5/"
lr_img_path3 =  "./ir_0.75/"
lr_img_path4 =  "./ir_1/"
lr_img_path5 =  "./vi_1/"


if os.path.exists(hr_img_path1) == False:
    print("wrong diretory")

if os.path.exists(lr_img_path1)  == False:
    os.makedirs(lr_img_path1)
if os.path.exists(lr_img_path2)  == False:
    os.makedirs(lr_img_path2)
if os.path.exists(lr_img_path3)  == False:
    os.makedirs(lr_img_path3)
if os.path.exists(lr_img_path4)  == False:
    os.makedirs(lr_img_path4)
if os.path.exists(lr_img_path5)  == False:
    os.makedirs(lr_img_path5)

file_name_list = os.listdir(hr_img_path1)

for name in file_name_list:
    img1 = Image.open(hr_img_path1+name)
    img2 = Image.open(hr_img_path2+name)
    h,w = img1.size
    h0,w0 = h//4*4,w//4*4
    img1 = Image.fromarray(np.array(img1)[:w0,:h0])
    img2 = Image.fromarray(np.array(img2)[:w0,:h0])

    # print(h,w)
    out = img1.resize((h//4,w//4))
    out.save(lr_img_path1+name)
    out = img1.resize((h//4*2,w//4*2))
    out.save(lr_img_path2+name)
    out = img1.resize((h//4*3,w//4*3))
    out.save(lr_img_path3+name)

    img1.save(lr_img_path4+name)
    img2.save(lr_img_path5+name)