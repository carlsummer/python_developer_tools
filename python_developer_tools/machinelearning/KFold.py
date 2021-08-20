# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/19/2021 4:41 PM
# @File:KFold
from sklearn.model_selection import KFold
import numpy as np

X = np.arange(24).reshape(12,2)
y = np.random.choice([1,2],12,p=[0.4,0.6])
kf = KFold(n_splits=5,shuffle=False) #shuffle=True,random_state=0
for train_index , test_index in kf.split(X):
   print('train_index:%s , test_index: %s ' %(train_index,test_index))

# import torch
# from sklearn.model_selection import KFold
#
# data_induce = np.arange(0, data_loader_old.dataset.length)
# kf = KFold(n_splits=5)
#
# for train_index, val_index in kf.split(data_induce):
#     train_subset = torch.utils.data.dataset.Subset(Dataset(params), train_index)
#     val_subset = torch.utils.data.dataset.Subset(Dataset(params), val_index)
#     data_loaders['train'] = torch.utils.data.DataLoader(train_subset, ...)
#     data_loaders['val'] = data.pair_provider_subset(val_subset, ...)

