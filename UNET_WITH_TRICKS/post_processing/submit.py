import tifffile
import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology


pred_dir = '/data/ss/URISC/CODE/FullVersion/exp/result/ensemble/'

pred_path = os.path.join(pred_dir, "*.png")

pred_list = glob(pred_path)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

for path in pred_list:
    pred = cv2.imread(path, 0)
    pred[pred < 20] = 0
    # plt.subplot(121)
    # plt.imshow(pred)
    # pred = cv2.dilate(pred, kernel)
    # plt.subplot(122)
    # plt.imshow(pred)  # Eroded Image
    # plt.show()


    save_dir = pred_dir + 'result/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    save_path = path.replace('.png', '.tiff')
    save_path = save_dir + os.path.split(save_path)[-1]
    print(save_path)
    cv2.imwrite(save_path, pred)

