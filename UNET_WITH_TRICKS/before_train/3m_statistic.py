import os
import numpy as np
from imutils import paths
from glob import glob
import cv2


def mean_max_min_statistic(root_dir, channels=3):
    train_image_path = os.path.join(root_dir, "IMAGE", "train", "*.png")
    train_image_list = glob(train_image_path)

    imgs_mean = [0 for i in range(channels)]
    imgs_max = [0 for i in range(channels)]
    imgs_min = [0 for i in range(channels)]

    for image_path in train_image_list:
        image = cv2.imread(image_path)

        image = image / 255.0

        img_mean = np.mean(image, axis=(0, 1))
        img_max = np.max(image, axis=(0, 1))
        img_min = np.min(image, axis=(0, 1))

        imgs_mean = imgs_mean + img_mean

        for i in range(channels):
            if imgs_max[i] < img_max[i]:
                imgs_max[i] = img_max[i]

        for i in range(channels):
            if imgs_min[i] > img_min[i]:
                imgs_min[i] = img_min[i]

    imgs_mean = imgs_mean / len(train_image_list)
    return imgs_mean, imgs_max, imgs_min


mean, max, min = mean_max_min_statistic(root_dir='E:/URISC/DATASET/', channels=3)
print(mean)
print(max)
print(min)