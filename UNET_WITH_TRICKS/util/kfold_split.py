import os
from glob import glob
from sklearn.model_selection import KFold
import numpy as np


def k_fold_split(train_dir, save_dir, k=5, save=True):
    train_val_image_path = os.path.join(train_dir, "*.png")
    train_val_image_list = np.array(glob(train_val_image_path))
    train_val_label_list = np.array([image_path.replace('IMAGE', 'LABEL') for image_path in train_val_image_list])

    kfold = KFold(n_splits=k, shuffle=True, random_state=47)

    if save:
        train_save_dir = os.path.join(save_dir, 'train/')
        val_save_dir = os.path.join(save_dir, 'val/')

        if not os.path.exists(train_save_dir):
            os.makedirs(train_save_dir)

        if not os.path.exists(val_save_dir):
            os.makedirs(val_save_dir)


    train_kfolds = []
    val_kfolds= []
    for idx, (train_index, val_index) in enumerate(kfold.split(train_val_image_list)):
        train_image_list = train_val_image_list[train_index]
        val_image_list = train_val_image_list[val_index]
        train_label_list = train_val_label_list[train_index]
        val_label_list = train_val_label_list[val_index]

        train_image_label_list = tuple(zip(train_image_list, train_label_list))

        val_image_label_list = tuple(zip(val_image_list, val_label_list))

        if save:
            train_txt_path = train_save_dir + 'train' + str(idx)
            train = open(train_txt_path, 'w')
            for image_path, label_path in zip(train_image_list, train_label_list):
                train.write(str(image_path) + ' ' + str(label_path) + '\n')
            train.close()

            val_txt_path = val_save_dir + 'train' + str(idx)
            val = open(val_txt_path, 'w')
            for image_path, label_path in zip(val_image_list, val_label_list):
                val.write(str(image_path) + ' ' + str(label_path) + '\n')
            val.close()

        train_kfolds.append(train_image_label_list)
        val_kfolds.append(val_image_label_list)
    return train_kfolds, val_kfolds
