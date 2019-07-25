from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import keras.backend as K
from utils.loss import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test(model_path, img_path):
    model = load_model(model_path, custom_objects={'iou': iou})

    image = tifffile.imread(img_path)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    soft_pred = model.predict(image)
    soft_pred = np.squeeze(soft_pred)

    hard_pred = K.round(soft_pred)

    return soft_pred, hard_pred


soft_pred, hard_pred = test(model_path='E:/Semantic-Segmentation/four_dataset/cvpr/EXPERIMENTS/1.vaihingen_ours_3.2/train/stack_decoder.hdf5',
                            img_path='E:/Semantic-Segmentation/four_dataset/cvpr/EXPERIMENTS/1.vaihingen_ours_3.2/test/23.tif')

plt.subplot(121)
plt.imshow(soft_pred)
plt.subplot(122)
plt.imshow(hard_pred)
plt.show()