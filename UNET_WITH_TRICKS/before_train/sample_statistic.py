import os
import numpy as np
from imutils import paths
from glob import glob
import cv2


def label_class_statistic(root_dir, class_value):
    train_label_path = os.path.join(root_dir, "LABEL", "train", "*.png")

    train_label_list = glob(train_label_path)

    # 生成与类别列表长度相同的字典，类别列表的值作为key，value全赋值为0
    value = [0 for i in range(0, len(class_value))]
    dict_ = dict(zip(class_value, value))  # 将key和value合成一个字典

    for label_path in train_label_list:
        label = cv2.imread(label_path, 0)

        # 循环遍历字典的key，并累加value值
        for key in dict_.keys():
            index = np.where(label == key)
            dict_[key] += len(index[0])
    return dict_


sample_statistic = label_class_statistic(root_dir='E:/URISC/DATASET/', class_value=[0, 255])
print(sample_statistic)

