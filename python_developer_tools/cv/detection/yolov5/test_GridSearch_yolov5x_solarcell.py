"""
搜索训练出来的模型的对于每个类别的最优的iou，conf，等参数
的值
"""
import argparse
import glob
import json
import os
import pickle
import random
import shutil
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    coco80_to_coco91_class, check_dataset, check_file, check_img_size, compute_loss, non_max_suppression_list,
    scale_coords,
    xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class, set_logging,
    print_mutation, plot_evolution, fitness)
from utils.torch_utils import select_device, time_synchronized


def test(batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6  # for NMS
         ):
    seen = 0
    s = ('%20s' + '%12s' * 6) % ('Class', labelname, "{}mAP".format(labelname), 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()

            # inf_out, train_out = model(img, augment=False)  # inference and training outputs
            modelOuputValDictkey = "key_{}".format(batch_i)
            if modelOuputValDict.__contains__(modelOuputValDictkey):
                inf_out, train_out = modelOuputValDict[modelOuputValDictkey]["inf_out"].to(device), \
                                     modelOuputValDict[modelOuputValDictkey]["train_out"]
            else:
                inf_out, train_out = model(img, augment=False)  # inference and training outputs
                modelOuputValDict[modelOuputValDictkey] = {"inf_out": inf_out.cpu(), "train_out": train_out}

            t0 += time_synchronized() - t

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression_list(inf_out, conf_thres_list=conf_thres, iou_thres=iou_thres, merge=merge)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='runs/exp2_yolov5x_solarcell/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/solarcell.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    modelOuputValDictpklpath = r"modelOuputValDict.pkl"

    # Initialize/load model and set device
    set_logging()
    device = select_device(opt.device, batch_size=opt.batch_size)
    merge, save_txt = opt.merge, opt.save_txt  # use Merge NMS, save *.txt labels

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    path = data['val']  # path to val/test images

    dataloader = create_dataloader(path, imgsz, opt.batch_size, model.stride.max(), opt,
                                   hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]

    modelOuputValDict = {}  # 用来存放所有模型预测出来的值
    if os.path.exists(modelOuputValDictpklpath):
        with open(modelOuputValDictpklpath, 'rb') as pkl_file:
            modelOuputValDict = pickle.load(pkl_file)

    best_param = {'prob': 0, 'real_width': 0, 'realHeight': 0, 'iou_thres': 0,
                  'iou_thres_mAP': 0, 'lubai_conf_thres': 0,
                  'lubai_conf_thres_mAP': 0, 'yiwu_conf_thres': 0.0,
                  'yiwu_conf_thres_mAP': 0, 'quejiao_conf_thres': 0,
                  'quejiao_conf_thres_mAP': 0, 'liepian_conf_thres': 0,
                  'liepian_conf_thres_mAP': 0, 'zangwu_conf_thres': 0,
                  'zangwu_conf_thres_mAP': 0}

    # 初始化conf_thres map
    hpy = {"lubai": {"thre_min": 0.0001, "thre_max": 0.0008, "thre_num": 40},
           "yiwu": {"thre_min": 0, "thre_max": 0.0011, "thre_num": 40},
           "quejiao": {"thre_min": 0, "thre_max": 0.01, "thre_num": 40},
           "liepian": {"thre_min": 0, "thre_max": 0.02, "thre_num": 180},
           "zangwu": {"thre_min": 0, "thre_max": 0.1, "thre_num": 180}}

    # for labelname in hpy.keys():
    #     best_param["{}_conf_thres".format(labelname)] = 0
    #     best_param["{}_conf_thres_mAP".format(labelname)] = 0

    # 如果随着搜索的值越大而mAP没有变大证明错过了最优阈值
    for iouthre in np.linspace(0, 1, 80):
        opt.iou_thres = iouthre
        # 寻找最好的confthres
        for labelname in hpy.keys():
            for labelconfs in np.linspace(hpy[labelname]["thre_min"], hpy[labelname]["thre_max"],
                                          hpy[labelname]["thre_num"]):
                labelconfthresKey = "{}_conf_thres".format(labelname)
                labelconfthresmapKey = "{}_conf_thres_mAP".format(labelname)

                opt.conf_thres = []
                mapsindex = 0
                for index, labelnametmp in enumerate(hpy.keys()):
                    labelconfthrestmpKey = "{}_conf_thres".format(labelnametmp)
                    if labelconfthresKey == labelconfthrestmpKey:
                        opt.conf_thres.append(labelconfs)
                        mapsindex = index
                    else:
                        opt.conf_thres.append(best_param[labelconfthrestmpKey])

                results, maps, times = test(opt.batch_size,
                                            opt.img_size,
                                            opt.conf_thres,
                                            opt.iou_thres)
                # 保存第一次运行完的结果
                if not os.path.exists(modelOuputValDictpklpath):
                    with open(modelOuputValDictpklpath, 'wb') as f:
                        pickle.dump(modelOuputValDict, f)

                if maps[mapsindex] > best_param[labelconfthresmapKey]:
                    best_param[labelconfthresKey] = labelconfs
                    best_param[labelconfthresmapKey] = maps[mapsindex]
                    print(best_param)
                if results[2] > best_param["iou_thres_mAP"]:
                    best_param["iou_thres"] = iouthre
                    best_param["iou_thres_mAP"] = results[2]
                    print(best_param)
    print('*' * 50)
    print(best_param)

# rm -rf nohup.out && nohup python test_scratch_hyp_yolov5x_solarcell.py & && tail -f nohup.out
