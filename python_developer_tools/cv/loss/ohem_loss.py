#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import ohem_cpp
from python_developer_tools.cv.loss.pytorchloss.pytorch_loss.large_margin_softmax import LargeMarginSoftmaxV3
import numpy as np

class OhemCELoss(nn.Module):

    def __init__(self, score_thresh, n_min=None, ignore_index=255):
        super(OhemCELoss, self).__init__()
        self.score_thresh = score_thresh
        self.ignore_lb = ignore_index
        self.n_min = n_min
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, logits, labels):
        n_min = labels.numel() // 16 if self.n_min is None else self.n_min
        labels = ohem_cpp.score_ohem_label(logits.float(), labels,
                self.ignore_lb, self.score_thresh, n_min).detach()
        loss = self.criteria(logits, labels)
        return loss


class OhemLargeMarginLoss(nn.Module):

    def __init__(self, score_thresh, n_min=None, ignore_index=255):
        super(OhemLargeMarginLoss, self).__init__()
        self.score_thresh = score_thresh
        self.ignore_lb = ignore_index
        self.n_min = n_min
        self.criteria = LargeMarginSoftmaxV3(
                ignore_index=ignore_index, reduction='mean')

    def forward(self, logits, labels):
        n_min = labels.numel() // 16 if self.n_min is None else self.n_min
        labels = ohem_cpp.score_ohem_label(logits.float(), labels,
                self.ignore_lb, self.score_thresh, n_min).detach()
        loss = self.criteria(logits, labels)
        return loss

# https://www.kaggle.com/c/bengaliai-cv19/discussion/128637
class OhemLoss(nn.Module):
    def __init__(self, rate=0.7):
        super(OhemLoss, self).__init__()
        self.rate = rate
        self.criteria = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    def forward(self, cls_pred, cls_target):
        batch_size = cls_pred.size(0)
        ohem_cls_loss = self.criteria(cls_pred, cls_target)

        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
        keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * self.rate))
        if keep_num < sorted_ohem_loss.size()[0]:
            keep_idx_cuda = idx[:keep_num]
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        cls_loss = ohem_cls_loss.sum() / keep_num
        return cls_loss


class NLL_OHEM(torch.nn.NLLLoss):
    """ Online hard example mining.
    Needs input from nn.LogSotmax() """

    def __init__(self, ratio):
        super(NLL_OHEM, self).__init__(None, True)
        self.ratio = ratio

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).cuda()
        for idx, label in enumerate(y.data):
            inst_losses[idx] = -x_.data[idx, label]
            # loss_incs = -x_.sum(1)
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return torch.nn.functional.nll_loss(x_hn, y_hn)

#------------------------------------------start cutmix mixup 数据增强对应用ohemloss------------------------------#
def ohem_loss(rate, cls_pred, cls_target):
    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    return data, targets


# loss
def cutmix_criterion(preds1, preds2, preds3, targets, rate=0.7):
    targets1, targets2, targets3, targets4, targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], \
                                                                      targets[4], targets[5], targets[6]
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion = ohem_loss
    return [lam * criterion(rate, preds1, targets1) + (1 - lam) * criterion(rate, preds1, targets2),
            lam * criterion(rate, preds2, targets3) + (1 - lam) * criterion(rate, preds2, targets4),
            lam * criterion(rate, preds3, targets5) + (1 - lam) * criterion(rate, preds3, targets6)]


def mixup(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]

    return data, targets


def mixup_criterion(preds1, preds2, preds3, targets, rate=0.7):
    targets1, targets2, targets3, targets4, targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], \
                                                                      targets[4], targets[5], targets[6]
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion = ohem_loss
    return [lam * criterion(rate, preds1, targets1) + (1 - lam) * criterion(rate, preds1, targets2),
            lam * criterion(rate, preds2, targets3) + (1 - lam) * criterion(rate, preds2, targets4),
            lam * criterion(rate, preds3, targets5) + (1 - lam) * criterion(rate, preds3, targets6)]
#------------------------------------------end cutmix mixup 数据增强对应用ohemloss------------------------------#

if __name__ == '__main__':
    criteria1 = OhemLargeMarginLoss(score_thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELoss(score_thresh=0.7, n_min=16*20*20//16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, 10, 10] = 255

    torch.autograd.set_detect_anomaly(True)

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear', align_corners=True)
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear', align_corners=True)

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs.clone())
    loss = loss1 + loss2
    loss.backward()


