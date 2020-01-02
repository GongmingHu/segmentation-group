import tifffile
import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

root_dir = 'E:/URISC/DATASET/'

train_label_path = os.path.join(root_dir, "LABEL", "train", "*.tiff")

train_label_list = glob(train_label_path)

for label_path in train_label_list:

    label = tifffile.imread(label_path)

    label_cp = label.copy()

    label[label_cp > 122] = 255
    label[label_cp < 122] = 0
    label[label_cp == 122] = 0

    count = len(np.where((label != 0) & (label != 255))[0])
    assert count == 0

    # plt.imshow(label)
    # plt.show()

    save_label_path = label_path.replace('.tiff', '.png')

    cv2.imwrite(save_label_path, label)

