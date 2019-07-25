# segmentation-group
CVAI语义分割组代码与数据共享仓库。

## 1. Keras-HRNet

### 代码组织形式

* `train.ipynb:`模型训练，包含超参设置、模型调用、训练、可视化。
* `test_crop_image.py:`模型测试，包含模型加载、测试、可视化。
* `dataloaders/generater.py:`数据加载，数据路径获取、图片读取、预处理及在线扩充。
* `model/seg_hrnet:`模型定义。
* `utils/loss.py:`损失函数，包含`dice_loss、ce_dice_loss、jaccard_loss(IoU loss)、ce_jaccard_loss、tversky_loss、focal_loss`
* `utils/metrics.py:`评价指标，包含`precision、recall、accuracy、iou、f1`等。