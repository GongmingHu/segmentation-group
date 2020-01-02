import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
import numpy as np
from glob import glob
import cv2


def random_crop(root_dir, window_size, random_num, image_save_path=None, label_save_path=None):
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)

    # 获取待裁剪影像和label的路径
    train_image_path = os.path.join(root_dir, "IMAGE", "train", "*.png")
    train_image_list = glob(train_image_path)

    i = 0
    for img_path in train_image_list:
        label_path = img_path.replace('IMAGE', 'LABEL')

        image = Image.open(img_path)
        label = Image.open(label_path)
        width = image.size[0]
        height = image.size[1]

        # 通过random.sample函数随机生成处于range范围之中的random_num个随机数，分别作为长和宽的起始坐标
        h_list = random.sample(range(0, height - window_size), random_num)
        w_list = random.sample(range(0, width - window_size), random_num)

        # 遍历长宽起始坐标列表,将原始图片随机裁剪为512大小
        for (h, w) in zip(h_list, w_list):
            # 获取裁剪横向和纵向起始和终止坐标
            slice_top = h
            slice_bottom = h + window_size
            slice_left = w
            slice_right = w + window_size

            # 使用Image的裁剪函数裁剪原始影像
            slice_img = image.crop((slice_left, slice_top, slice_right, slice_bottom))
            slice_img.save(image_save_path + str(i) + '.png')

            # 使用Image的裁剪函数裁剪label
            slice_label = label.crop((slice_left, slice_top, slice_right, slice_bottom))
            slice_label.save(label_save_path + str(i) + '.png')

            i = i + 1


random_crop(root_dir='E:/URISC/DATASET/',
            window_size=512,
            random_num=100,
            image_save_path='E:/URISC/DATASET/IMAGE/train_crop_with_complex/',
            label_save_path='E:/URISC/DATASET/LABEL/train_crop_with_complex/')