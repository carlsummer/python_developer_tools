# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/18/2021 4:48 PM
# @File:cross_entropy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from python_developer_tools.cv.loss.classes.LabelSmoothingCrossEntropy import LabelSmoothingCrossEntropy


def sigmoid(x):
    return (1 + (-x).exp()).reciprocal()


def binary_cross_entropy(pred, y):
    return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class JsdCrossEntropy(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss
    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    Hacked together by / Copyright 2020 Ross Wightman
    """
    def __init__(self, num_splits=3, alpha=12, smoothing=0.1):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        if smoothing is not None and smoothing > 0:
            self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = torch.split(output, split_size)

        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]

        # Clamp mixture distribution to avoid exploding KL divergence
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
        loss += self.alpha * sum([F.kl_div(
            logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / len(probs)
        return loss