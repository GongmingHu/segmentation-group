import os
import numpy as np
from imutils import paths
from glob import glob
import cv2



def std_statistic(root_dir, imgs_mean, channels=3):
    train_image_path = os.path.join(root_dir, "IMAGE", "train", "*.png")
    train_image_list = glob(train_image_path)

    imgs_std = [0 for i in range(channels)]

    for image_path in train_image_list:
        image = cv2.imread(image_path)
        image = image / 255.0

        img_std = (image - imgs_mean) ** 2  # [1024, 1024, 3]

        img_std = np.mean(img_std, axis=(0, 1))  # [3,]

        imgs_std = imgs_std + img_std

    imgs_std = imgs_std / len(train_image_list)
    imgs_std = np.sqrt(imgs_std)
    return imgs_std

std = std_statistic(root_dir='E:/URISC/DATASET/', imgs_mean=[0.52011591, 0.52011591, 0.52011591], channels=3)
print(std)