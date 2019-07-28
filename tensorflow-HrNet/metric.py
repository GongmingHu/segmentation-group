import numpy as np
from sklearn.metrics import accuracy_score
from PIL import Image

class Metric():
    def __init__(self):
        None

    def OA(self,gt,pred):
         OA = accuracy_score(np.reshape(gt, [-1]), np.reshape(pred, [-1]))
         return OA

    def recall(slef,gt,pred,class_which = None):
        #求TP
        gt_which_index = gt == class_which
        gt_notwhich_index = gt != class_which
        print(len(np.argwhere(gt == class_which)))
        gt[gt_which_index] = 1
        gt[gt_notwhich_index] = 0

        pred_which_index = pred == class_which
        pred_notwhich_index = pred != class_which
        pred[pred_which_index] = 1
        pred[pred_notwhich_index] = 0
        TP = gt * pred
        TP = len(np.argwhere(TP == 1))

        #求TP+TN
        TP_TN = len(np.argwhere(gt == 1))


        recall = TP / TP_TN
        return recall

    def precision(self,gt,pred,class_which = None):
        # 求TP
        gt_which_index = gt == class_which
        print(len(np.argwhere(gt == class_which)))
        gt_notwhich_index = gt != class_which
        gt[gt_which_index] = 1
        gt[gt_notwhich_index] = 0

        pred_which_index = pred == class_which
        pred_notwhich_index = pred != class_which
        pred[pred_which_index] = 1
        pred[pred_notwhich_index] = 0
        TP = gt * pred
        TP = len(np.argwhere(TP == 1))
        #print(TP)

        #求TP_FP
        TP_FP = len(np.argwhere(pred == 1))
        #print(TP_FP)

        precision = TP /TP_FP
        return precision

    def F1_score(self,label,pred,class_which = None):
        recall = self.recall(label,pred,class_which)
        precision = self.precision(label,pred,class_which)

        F1_score = 2*(recall*precision/(precision+recall))
        return F1_score







#metric = Metric()

# #打开图像并转化为灰度图像
# image =  Image.open("annotation.png").convert("L")
# #将图像转化为数组
# image_array = np.array(image)
#
# label =  Image.open("dataset/ISPRS 2D Vaihingen/prelabel/000.png").convert("L")
# #将图像转化为数组
# label_array = np.array(label)
# label_array = label_array[0:2560,0:1792]

#print(metric.OA(label_array,image_array))
#print(metric.recall(label_array,image_array,4))
#print(metric.precision(label_array,image_array,4))
#print(metric.F1_score(label_array,image_array,4))


