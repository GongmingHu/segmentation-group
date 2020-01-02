import os
import time
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from glob import glob

from util import dataset, transform, config
from util.util import AverageMeter, check_makedirs
from model.seg_model.unet.vgg13_unet import UNet
from model.seg_model.unet.resnet34_unet import adoptedUNet
from model.seg_model.hed.hed_model import HED
from util.metric import accuracy_metrics

cv2.ocl.setUseOpenCL(False)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='/data/ss/URISC/CODE/FullVersion/config/URISC/urisc_unet.yaml', help='config file')
    parser.add_argument('opts', help='see config/URISC/urisc_unet.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def check(args):
    pass


def main():
    # params parser
    global args, logger
    args = get_parser()
    # params check
    check(args)
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    # ----------------- Test setting ----------------- #
    # load model
    if args.arch == 'unet':
        model = UNet(n_classes=args.classes, bilinear=args.bilinear_up).cuda()
    elif args.arch == 'resnet_unet':
        model = adoptedUNet(layer=34, use_ppm=True, use_attention=False,
                            up_way=args.upway, num_classes=args.classes).cuda()
    elif args.arch == 'hed':
        model = HED().cuda()
    logger.info(model)
    if len(args.train_gpu) > 1:
        model = torch.nn.DataParallel(model)
    cudnn.benchmark = False

    # ----------------- data loader ----------------- #
    value_scale = 255
    mean = args.mean
    mean = [item * value_scale for item in mean]
    std = args.std
    std = [item * value_scale for item in std]

    test_transform = transform.Compose([
        # transform.RandomBilateralFilter(p=1),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    test_image_path = os.path.join(args.test_image_dir, "*.png")
    test_image_list = glob(test_image_path)
    test_image_list = tuple(zip(test_image_list, test_image_list))
    test_data = dataset.SemData(split=args.split, data_root=args.data_root,
                                data_list=test_image_list, transform=test_transform)
    index_start = args.index_start
    if args.index_step == 0:
        index_end = len(test_data.data_list)
    else:
        index_end = min(index_start + args.index_step, len(test_data.data_list))
    test_data.data_list = test_data.data_list[index_start:index_end]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    if len(args.model_path) != 0:

        for fold_i in range(args.folds):
            for model_i in args.model_path:
                single_model_path = args.model_save_dir + 'Fold{}/train_epoch_{}.pth'.format(fold_i, model_i)
                single_save_folder = args.result_save_dir + 'Fold{}/epoch_{}/'.format(fold_i, model_i)
                if os.path.isfile(single_model_path):
                    logger.info("=> loading checkpoint '{}'".format(single_model_path))
                    checkpoint = torch.load(single_model_path)
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                    logger.info("=> loaded checkpoint '{}'".format(single_model_path))
                else:
                    raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
                # test(test_loader, test_data.data_list, model, args.classes, args.base_size,
                #      args.test_h, args.test_w, args.scales, single_save_folder)

    if len(args.model_path) != 0:
        ensemble(test_data.data_list, args.base_size, args.base_size, args.ensemble_way, args.threshold)

    if args.split != 'test':
        cal_acc(test_data.data_list, args.ensemble_folder, args.classes)


def net_process(model, image, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)

    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, stride_rate=1/2):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def test(test_loader, data_list, model, classes, base_size, crop_h, crop_w, scales, binary_folder):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, _) in enumerate(test_loader):
        data_time.update(time.time() - end)
        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w)

        prediction /= len(scales)

        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        check_makedirs(binary_folder)
        image_path, _ = data_list[i]
        image_name = os.path.split(image_path)[-1]
        image_name = image_name.replace('png', 'npy')
        save_path = os.path.join(binary_folder, image_name)
        np.save(save_path, prediction)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def ensemble(data_list, h, w, ensemble_way='mean_of_mean_and_max', threshold=0.85):
    for i, (image_path, target_path) in enumerate(data_list):
        image_name = os.path.split(image_path)[-1]
        pred_npy_name = image_name.replace('png', 'npy')
        prediction = np.zeros((h, w, args.folds*len(args.model_path)), dtype=float)
        for fold_i in range(args.folds):
            for idx, model_i in enumerate(args.model_path):
                single_npy_path = args.result_save_dir + 'Fold{}/epoch_{}/'.format(fold_i, model_i) + pred_npy_name
                pred_npy = np.load(single_npy_path)
                prediction[:, :, fold_i*len(args.model_path)+idx] = pred_npy

        if ensemble_way == 'mean':
            prediction = np.mean(prediction, axis=-1)
        elif ensemble_way == 'max':
            prediction = np.max(prediction, axis=-1)
        elif ensemble_way == 'mean_of_mean_and_max':
            mean_prediction = np.mean(prediction, axis=-1)
            max_prediction = np.max(prediction, axis=-1)
            final_prediction = np.zeros((h, w, 2), dtype=float)
            final_prediction[:, :, 0] = mean_prediction
            final_prediction[:, :, 1] = max_prediction
            prediction = np.mean(final_prediction, axis=-1)

        prediction_cp = prediction.copy()
        prediction[(prediction_cp > threshold) & (prediction_cp == threshold)] = 0
        prediction[prediction_cp < threshold] = 255

        ensemble_folder = args.result_save_dir + 'ensemble/'
        check_makedirs(ensemble_folder)
        binary = np.uint8(prediction)
        binary_path = os.path.join(ensemble_folder, image_name)
        cv2.imwrite(binary_path, binary)

    logger.info('save ensemble file finish!')


def cal_acc(data_list, pred_folder, classes):
    accuracy_meter = AverageMeter()
    fscore_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = os.path.split(image_path)[-1]
        pred = cv2.imread(os.path.join(pred_folder, image_name), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        accuracy, precision, recall, f_score = accuracy_metrics(pred, target)
        accuracy_meter.update(accuracy)
        fscore_meter.update(f_score)

        logger.info('Evaluating {0}/{1} on image {2}, fscore {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', fscore_meter.val))

    mAcc = accuracy_meter.avg
    mFscore = fscore_meter.avg
    logger.info('Eval result: mAcc/mFscore {:.4f}/{:.4f}.'.format(mAcc/mFscore))


if __name__ == '__main__':
    main()
