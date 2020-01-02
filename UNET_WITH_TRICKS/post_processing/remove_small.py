import numpy as np
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt
from skimage import morphology


def remove_small_area(image, min_size):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img2 = np.zeros(output.shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 1

    return img2


pred_dir = 'E:/URISC/CODE/UNET/exp/urisc/unet/result/ensemble/'

pred_path = os.path.join(pred_dir, "*.png")

pred_list = glob(pred_path)

for path in pred_list:
    pred = cv2.imread(path, 0)

    pred_cp = pred.copy()
    pred[pred_cp == 255] = 0
    pred[pred_cp == 0] = 1

    out = remove_small_area(pred, 10)

    out_cp = out.copy()

    out[out_cp == 1] = 0
    out[out_cp == 0] = 255

    # plt.subplot(121)
    # plt.imshow(pred)
    # plt.subplot(122)
    # plt.imshow(out)
    # plt.show()

    save_path = path
    cv2.imwrite(save_path, out)



