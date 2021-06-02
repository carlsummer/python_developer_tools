"""
搜索训练出来的模型的对于每个类别的最优的iou，conf，等参数
的值
rm -rf nohup.out
nohup python test_hyperopt_yolov5x_solarcell.py &
tail -f nohup.out
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

import hyperopt
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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from collections import OrderedDict
from models.yolo import Model


def test(batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         widthHeights={},
         ):
    seen = 0
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

            lhpred = pred.cpu().numpy()
            predLabels = lhpred[:, 5]
            predconfthres = lhpred[:, 4]
            predXWidths = lhpred[:, 2] - lhpred[:, 0]
            predYHeights = lhpred[:, 3] - lhpred[:, 1]
            # 获取各个类别width，height 的最大最小值
            if not os.path.exists(modelOuputValDictpklpath):
                for labelindex, labelname in enumerate(data["names"]):
                    labelconfthrelist = predconfthres[predLabels == labelindex]
                    if len(labelconfthrelist) > 0:
                        labelwidthpredlist = predXWidths[predLabels == labelindex]
                        labelheightpredlist = predYHeights[predLabels == labelindex]

                        confthresmin = float(labelconfthrelist.min())
                        confthresmax = float(labelconfthrelist.max())
                        widthmin = float(labelwidthpredlist.min())
                        widthmax = float(labelwidthpredlist.max())
                        heightmin = float(labelheightpredlist.min())
                        heightmax = float(labelheightpredlist.max())

                        confthresminkey = "{}_confthres_min".format(labelname)
                        confthresmaxkey = "{}_confthres_max".format(labelname)
                        widthminkey = "{}_width_min".format(labelname)
                        widthmaxkey = "{}_width_max".format(labelname)
                        heightminkey = "{}_height_min".format(labelname)
                        heightmaxkey = "{}_height_max".format(labelname)

                        if not modelOuputValDict.__contains__(confthresminkey) or confthresmin < modelOuputValDict[
                            confthresminkey]:
                            modelOuputValDict[confthresminkey] = confthresmin
                        if not modelOuputValDict.__contains__(confthresmaxkey) or confthresmax > modelOuputValDict[
                            confthresmaxkey]:
                            modelOuputValDict[confthresmaxkey] = confthresmax
                        if not modelOuputValDict.__contains__(widthminkey) or widthmin < modelOuputValDict[widthminkey]:
                            modelOuputValDict[widthminkey] = widthmin
                        if not modelOuputValDict.__contains__(widthmaxkey) or widthmax > modelOuputValDict[widthmaxkey]:
                            modelOuputValDict[widthmaxkey] = widthmax
                        if not modelOuputValDict.__contains__(heightminkey) or heightmin < modelOuputValDict[
                            heightminkey]:
                            modelOuputValDict[heightminkey] = heightmin
                        if not modelOuputValDict.__contains__(heightmaxkey) or heightmax > modelOuputValDict[
                            heightmaxkey]:
                            modelOuputValDict[heightmaxkey] = heightmax
            else:
                # 搜索超参数 width，height 用的过滤规则
                widthHeightFilterList = []
                for widthheightkey in widthHeights.keys():
                    widthHeightValList = widthHeights[widthheightkey]
                    widthHeightFilterListtmp = (predLabels == int(widthheightkey)) & (
                            predXWidths >= widthHeightValList[0]) & (
                                                       predXWidths <= widthHeightValList[1]) & (
                                                       predYHeights >= widthHeightValList[2]) & (
                                                       predYHeights <= widthHeightValList[3])
                    if len(widthHeightFilterList) == 0:
                        widthHeightFilterList = widthHeightFilterListtmp
                    else:
                        widthHeightFilterList = widthHeightFilterList | widthHeightFilterListtmp
                pred = pred[widthHeightFilterList]

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
    parser.add_argument('--weights', nargs='+', type=str, default='runs1/hn_9bb3c_danjing_bp_solarcell.pt',
                        help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='runs1/data.yaml', help='*.data path')
    parser.add_argument('--ymal', type=str, default='runs1/hn_9bb3c_solarcell.yaml', help='*.data path')
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

    modelOuputValDictpklpath = r"modelOuputValDict2.pkl"

    # Initialize/load model and set device
    set_logging()
    device = select_device(opt.device, batch_size=opt.batch_size)
    merge, save_txt = opt.merge, opt.save_txt  # use Merge NMS, save *.txt labels

    # Load model
    # model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    # imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
    # -----
    model = Model(cfg=opt.ymal)
    state_dictBA = torch.load(opt.weights, map_location='cpu')['model']
    new_state_dictBA = OrderedDict()
    for k, v in state_dictBA.items():
        name = k[7:]  # remove `module.`
        new_state_dictBA[name] = v
    model.load_state_dict(new_state_dictBA)
    model.float().fuse().eval()
    model = model.to(device)

    imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
    # -----

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
    path = data[opt.task]  # path to val/test images

    dataloader = create_dataloader(path, imgsz, opt.batch_size, model.stride.max(), opt,
                                   hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]

    # 获取模型预测出来的值
    modelOuputValDict = {}  # 用来存放所有模型预测出来的值
    if os.path.exists(modelOuputValDictpklpath):
        with open(modelOuputValDictpklpath, 'rb') as pkl_file:
            modelOuputValDict = pickle.load(pkl_file)
    else:
        # 初始化要搜索的超参数的值
        for labelindex, labelname in enumerate(data["names"]):
            confthresminkey = "{}_confthres_min".format(labelname)
            confthresmaxkey = "{}_confthres_max".format(labelname)
            widthminkey = "{}_width_min".format(labelname)
            widthmaxkey = "{}_width_max".format(labelname)
            heightminkey = "{}_height_min".format(labelname)
            heightmaxkey = "{}_height_max".format(labelname)

            modelOuputValDict[confthresminkey]=0
            modelOuputValDict[confthresmaxkey]=1
            modelOuputValDict[widthminkey]=0
            modelOuputValDict[widthmaxkey]=1
            modelOuputValDict[heightminkey]=0
            modelOuputValDict[heightmaxkey]=1


        results, maps, times = test(opt.batch_size, opt.img_size,
                                    [0, 0, 0, 0, 0], 0, {})
        # 保存第一次运行的值
        with open(modelOuputValDictpklpath, 'wb') as f:
            pickle.dump(modelOuputValDict, f)

    # 开始超参数搜索
    best_params ={"best_mAP":0}

    def findhpye(params):
        global data

        widthheightlist = {}
        conf_thres = []
        for labelindex, labelname in enumerate(data["names"]):
            conf_thres.append(params["{}_conf_thres".format(labelname)])
            widthheightlist[str(labelindex)] = [params["{}_realWidth_min".format(labelname)],
                                                params["{}_realWidth_max".format(labelname)],
                                                params["{}_realHeight_min".format(labelname)],
                                                params["{}_realHeight_max".format(labelname)]]
        results, maps, times = test(params["batch_size"], params["img_size"],
                                    conf_thres,
                                    params["iou_thres"],
                                    widthheightlist
                                    )
        # Extract the best score
        mAP = results[2]

        global best_params
        if mAP > best_params["best_mAP"]:
            best_params = params
            best_params["best_mAP"] = mAP
            print(best_params)
            with open("best_params.txt", 'w', encoding='utf-8') as f:
                f.write(str(best_params))
        # Loss must be minimized
        loss = 1 - mAP

        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'status': STATUS_OK}


    space4knn = {
        "batch_size": opt.batch_size,
        "img_size": opt.img_size,
        'iou_thres': hp.choice('iou_thres', np.linspace(0, 1, 100)),
        #'iou_thres': hp.uniform('iou_thres', 0,1), #参数在low和high之间均匀分布。
    }

    for labelindex, labelname in enumerate(data["names"]):
        confthresminkey = "{}_confthres_min".format(labelname)
        confthresmaxkey = "{}_confthres_max".format(labelname)
        widthminkey = "{}_width_min".format(labelname)
        widthmaxkey = "{}_width_max".format(labelname)
        heightminkey = "{}_height_min".format(labelname)
        heightmaxkey = "{}_height_max".format(labelname)

        space4knnconfthrekey = "{}_conf_thres".format(labelname)
        space4knnrealWidth_minkey = "{}_realWidth_min".format(labelname)
        space4knnrealWidth_maxkey = "{}_realWidth_max".format(labelname)
        space4knnrealHeight_minkey = "{}_realHeight_min".format(labelname)
        space4knnrealHeight_maxkey = "{}_realHeight_max".format(labelname)

        space4knn[space4knnconfthrekey] = hp.choice(space4knnconfthrekey,
                                                    np.linspace(modelOuputValDict[confthresminkey],
                                                                modelOuputValDict[confthresmaxkey], 1000))
        # space4knn[space4knnconfthrekey] = hp.uniform(space4knnconfthrekey,modelOuputValDict[confthresminkey],modelOuputValDict[confthresmaxkey])
        space4knn[space4knnrealWidth_minkey] = hp.choice(space4knnrealWidth_minkey,
                                                         np.linspace(0, int(modelOuputValDict[widthminkey]),
                                                                     1 if int(modelOuputValDict[widthminkey]) == 0 else int(modelOuputValDict[widthminkey]), dtype=int))
        space4knn[space4knnrealWidth_maxkey] = hp.choice(space4knnrealWidth_maxkey,
                                                         np.linspace(int(modelOuputValDict[widthminkey]),
                                                                     int(modelOuputValDict[widthmaxkey]),
                                                                     int(modelOuputValDict[widthmaxkey] -
                                                                         modelOuputValDict[widthminkey]), dtype=int))
        space4knn[space4knnrealHeight_minkey] = hp.choice(space4knnrealHeight_minkey,
                                                          np.linspace(0, int(modelOuputValDict[heightminkey]),
                                                                      1 if int(modelOuputValDict[heightminkey]) == 0 else int(modelOuputValDict[heightminkey]), dtype=int))
        space4knn[space4knnrealHeight_maxkey] = hp.choice(space4knnrealHeight_maxkey,
                                                          np.linspace(int(modelOuputValDict[heightminkey]),
                                                                      int(modelOuputValDict[heightmaxkey]),
                                                                      int(modelOuputValDict[heightmaxkey] -
                                                                          modelOuputValDict[heightminkey]), dtype=int))

    trials = Trials()

    """
    随机搜索(对应是hyperopt.rand.suggest)，
    模拟退火(对应是hyperopt.anneal.suggest)，
    对应是hyperopt.TPE.suggest算法 Tree-structured Parzen Estimator Approach
    # define an algorithm that searches randomly 5% of the time,
    # uses TPE 75% of the time, and uses annealing 20% of the time
    mix_algo = partial(mix.suggest, p_suggest=[
    (0.05, rand.suggest),
    (0.75, tpe.suggest),
    (0.20, anneal.suggest)])
    estim = HyperoptEstimator(algo=mix_algo,
    max_evals=150,
    trial_timeout=60)
    """
    best = fmin(findhpye, space4knn, algo=hyperopt.anneal.suggest, max_evals=1000000, trials=trials)
    print("最好的参数index{}".format(best))
    # bestsample = hyperopt.pyll.stochastic.sample(space4knn) 只是打印的样例
    # 将搜索到的最好的参数转换为对应的值
    bestValue = {}
    for bestkey in best.keys():
        if len(space4knn[bestkey].pos_args) == 1:
            bestValue[bestkey] = best[bestkey]
        else:
            bestValue[bestkey] = space4knn[bestkey].pos_args[best[bestkey] + 1].obj
    print("最好的参数val{}".format(bestValue))
    # 最后获取搜索出来最优的参数的mAP的值
    widthheightlist = {}
    conf_thres = []
    for labelindex, labelname in enumerate(data["names"]):
        conf_thres.append(bestValue["{}_conf_thres".format(labelname)])
        widthheightlist[str(labelindex)] = [bestValue["{}_realWidth_min".format(labelname)],
                                            bestValue["{}_realWidth_max".format(labelname)],
                                            bestValue["{}_realHeight_min".format(labelname)],
                                            bestValue["{}_realHeight_max".format(labelname)]]
    results, maps, times = test(opt.batch_size, opt.img_size,
                                conf_thres,
                                bestValue["iou_thres"],
                                widthheightlist)
    print(results[2])
# rm -rf nohup.out && nohup python test_scratch_hyp_yolov5x_solarcell.py & && tail -f nohup.out
