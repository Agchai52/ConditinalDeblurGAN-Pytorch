"""
Load and Save a Single Image
"""
from __future__ import division
import numpy as np
import math
from PIL import Image
import os


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename+'.png')


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def find_latest_model(net_g_path):
    file_list = os.listdir(net_g_path)
    model_names = [f[13:-4] for f in file_list if ".pth" in f]
    iter_num = max(model_names)
    return os.path.join(net_g_path, "G_model_step_{}.pth".format(iter_num))
