import os
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

from util import dataset, transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU
from util.metric import accuracy_metrics
from model.seg_model.unet.vgg13_unet import UNet
from model.seg_model.unet.resnet34_unet import adoptedUNet
from model.seg_model.hed.hed_model import HED
from util.loss import *
from util.kfold_split import *
from holocron.optim import RAdam, Lookahead

cv2.ocl.setUseOpenCL(False)  # 这行代码应该是禁用opencl，opencl是一个GPU的加速技术
cv2.setNumThreads(0)  # 该函数设置opencv的线程数量，参数为0时，则仅进行单线程运算


# ----------------- params parser ----------------- #
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation') # 创建 ArgumentParser() 对象
    # 添加参数：--表示可选参数，否则为定位参数
    parser.add_argument('--config', type=str, default='/data/ss/URISC/CODE/FullVersion/config/URISC/urisc_unet.yaml', help='config file')
    parser.add_argument('opts', help='see config/URISC/urisc_unet.yaml for all options', default=None, nargs=argparse.REMAINDER)
    # 解析添加的参数
    args = parser.parse_args()
    assert args.config is not None
    # 将yaml文件加载为CfgNode
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


# ----------------- params check ----------------- #
def check(args):
    pass


# ----------------- log ----------------- #
def get_logger():
    logger_name = "main-logger"
    # Logger是一个树形层级结构，在使用接口debug，info，warn，error，critical之前必须创建Logger实例，即创建一个记录器，
    logger = logging.getLogger(logger_name)
    # 设置日志级别为INFO，即只有日志级别大于等于INFO的日志才会输出
    logger.setLevel(logging.INFO)
    # 创建一个处理器
    handler = logging.StreamHandler()
    # 使用Formatter对象设置日志信息最后的规则、结构和内容
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    # 为Logger实例增加一个处理器
    logger.addHandler(handler)
    return logger

def main():
    # params parser
    global args, writer, logger
    args = get_parser()

    logger = get_logger()
    logger.info(args)
    logger.info("Classes: {}".format(args.classes))
    # params check
    check(args)
    # params set
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    # set random number
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    # ----------------- data preprocessing ----------------- #
    value_scale = 255
    mean = args.mean
    mean = [item * value_scale for item in mean]
    std = args.std
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean),
        transform.RandomHorizontalFlip(),
        transform.RandomBilateralFilter(p=0.5),
        transform.RandomElastic(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    val_transform = transform.Compose([
        # transform.RandomBilateralFilter(p=1),
        # transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    # split train & val
    train_kfolds, val_kfolds = k_fold_split(train_dir=args.train_image_dir,
                                            save_dir=args.txt_save_dir,
                                            k=args.folds, save=True)
    for fold_i, (train_image_label_list, val_image_label_list) in enumerate(zip(train_kfolds, val_kfolds)):
        print('>>>>>>>>>>>>>>>> Start Fold {} >>>>>>>>>>>>>>>>'.format(fold_i))
        # ----------------- Train setting ----------------- #
        # loss
        if args.loss == 'wbce':
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.edge_weight))
        elif args.loss == 'dilatedbce':
            criterion = dilatedweightBCE(kernel_size=3, bg_weight=args.bg_weight,
                                         dilated_bg_weight=args.dilated_bg_weight,
                                         edge_weight=args.edge_weight)
        elif args.loss == 'focal':
            criterion = FocalLoss(alpha=1, gamma=2, logits=True, weight=args.edge_weight, reduce=True)
        elif args.loss == 'dice':
            criterion = DiceLoss()
        elif args.loss == 'focal_dice':
            criterion = FocalDiceLoss(alpha=1, gamma=2, logits=True, weight=args.edge_weight, reduce=True)

        # model
        if args.arch == 'unet':
            model = UNet(n_classes=args.classes, bilinear=args.bilinear_up, criterion=criterion).cuda()
        elif args.arch == 'resnet_unet':
            model = adoptedUNet(layer=34, use_ppm=True, use_attention=False,
                                up_way=args.upway, num_classes=args.classes,
                                pretrained=True, criterion=criterion).cuda()
        elif args.arch == 'hed':
            model = HED(criterion=criterion).cuda()
        logger.info(model)

        # model parallel
        if len(args.train_gpu) > 1:
            logger.info("%d GPU parallel" % len(args.train_gpu))
            model = nn.DataParallel(model)

        # optimizer
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=args.weight_decay, amsgrad=False)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        elif args.optimizer == 'radam':
            optimizer = RAdam(model.parameters(), lr=args.base_lr)
            # Wrap it with Lookahead
            optimizer = Lookahead(optimizer, sync_rate=0.5, sync_period=6)

        # checkpoint resume
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
                args.start_epoch = checkpoint['epoch']
                model_dict = model.state_dict()
                old_dict = {k: v for k, v in checkpoint['state_dict'].items() if (k in model_dict)}
                model_dict.update(old_dict)
                model.load_state_dict(model_dict)

                # model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

        # ---------------------- data loader ---------------------------- #
        train_image_label_list = train_image_label_list*100
        save_path = os.path.join(args.model_save_dir, ('Fold' + str(fold_i)))
        global writer
        writer = SummaryWriter(save_path)
        # data loader for training
        train_data = dataset.SemData(split='train', data_root=args.data_root, data_list=train_image_label_list,
                                     transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.workers,
                                                   pin_memory=True, sampler=None, drop_last=True)
        logger.info("Train set: %d" % (len(train_data)))

        # data loader for validation
        if args.evaluate:
            val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=val_image_label_list,
                                       transform=val_transform)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                     shuffle=False, num_workers=args.workers,
                                                     pin_memory=True, sampler=None)
            logger.info("val set: %d" % (len(val_data)))

        # ----------------- Train and Val ----------------- #
        for epoch in range(args.start_epoch, args.epochs):
            epoch_log = epoch + 1
            # train
            loss_train, mAcc_train, mFscore_train = train(train_loader, model, optimizer, epoch)
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('mFscore_train', mFscore_train, epoch_log)

            # save model
            if epoch_log % args.save_freq == 0:

                filename = save_path + '/train_epoch_' + str(epoch_log) + '.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
                if epoch_log / args.save_freq > 20:
                    deletename = save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 20) + '.pth'
                    os.remove(deletename)

            # val
            if args.evaluate:
                with torch.no_grad():
                    loss_val, mAcc_val, mFscore_val, max_threshold = validate(val_loader, model, criterion)
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('mFscore_val', mFscore_val, epoch_log)
                writer.add_scalar('max_threshold', max_threshold, epoch_log)


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    fscore_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # data
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # loss
        logits, output, loss = model(input, target)
        if len(args.train_gpu) > 1:
            loss = torch.mean(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = input.size(0)
        loss_meter.update(loss.item(), n)

        # metric
        accuracy, precision, recall, f_score = accuracy_metrics(output.detach().cpu().numpy(),
                                                                target.detach().cpu().numpy(),
                                                                threshold=args.binary_threshold,
                                                                training=True)
        accuracy_meter.update(accuracy)
        fscore_meter.update(f_score)

        batch_time.update(time.time() - end)
        end = time.time()

        # learning rate
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        writer.add_scalar('learning_rate', current_lr, current_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy_meter.val:.4f}.'
                        'f-score {fscore_meter.val:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                             batch_time=batch_time,
                                                             data_time=data_time,
                                                             remain_time=remain_time,
                                                             loss_meter=loss_meter,
                                                             accuracy_meter=accuracy_meter,
                                                             fscore_meter=fscore_meter))

        writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
        writer.add_scalar('accuracy_train_batch', accuracy_meter.val, current_iter)
        writer.add_scalar('fscore_train_batch', fscore_meter.val, current_iter)

    mAcc = accuracy_meter.avg
    mFscore = fscore_meter.avg
    logger.info('Train result at epoch [{}/{}]: mAcc/mFscore {:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mAcc, mFscore))
    return loss_meter.avg, mAcc, mFscore


def validate(val_loader, model, criterion):
    print('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    fscore_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        loss = criterion(output, target)

        n = input.size(0)
        loss = torch.mean(loss)

        # metric
        accuracy, precision, recall, f_score, max_threshold = accuracy_metrics(output.detach().cpu().numpy(),
                                                                target.detach().cpu().numpy(),
                                                                threshold=args.binary_threshold,
                                                                training=False)
        accuracy_meter.update(accuracy)
        fscore_meter.update(f_score)

        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0:
            print('Test: [{}/{}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                  'Accuracy {accuracy_meter.val:.4f}.'
                  'Fscore {fscore_meter.val:.4f}.'
                  'max_threshold {max_threshold:.2f}'.format(i + 1, len(val_loader),
                                                    data_time=data_time,
                                                    batch_time=batch_time,
                                                    loss_meter=loss_meter,
                                                    accuracy_meter=accuracy_meter,
                                                    fscore_meter=fscore_meter,
                                                    max_threshold=max_threshold))

    mAcc = accuracy_meter.avg
    mFscore = fscore_meter.avg
    print('Val result: mAcc/mFscore {:.4f}/{:.4f}.'.format(mAcc, mFscore))
    print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mAcc, mFscore, max_threshold


if __name__ == '__main__':
    main()
