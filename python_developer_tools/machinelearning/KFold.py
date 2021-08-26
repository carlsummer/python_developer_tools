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
"""https://discuss.pytorch.org/t/i-need-help-in-this-k-fold-cross-validation-implementation/90705"""
# total_set  = datasets.ImageFolder(root_dir)
# splits = KFold(n_splits = 5, shuffle = True, random_state = 42)
# .............
# for fold, (train_idx, valid_idx) in enumerate(splits.split(total_set)):
#     print('Fold : {}'.format(fold))
#     train_sampler = SubsetRandomSampler(train_idx)
#     valid_sampler = SubsetRandomSampler(valid_idx)
#     train_loader = torch.utils.data.DataLoader(
#                       WrapperDataset(total_set,  transform=transforms['train']),
#                       batch_size=64, sampler=train_sampler)
#     valid_loader = torch.utils.data.DataLoader(
#                       WrapperDataset(total_set, transform = transforms['valid']),
#                       batch_size=64, sampler=valid_sampler)
#     model.load_state_dict(model_wts)
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         running_corrects = 0
#         trunning_corrects = 0
#         for inputs, labels in train_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#             with torch.set_grad_enabled(True):
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += (preds == labels).sum()
#             trunning_corrects += preds.size(0)
#             scheduler.step()
#
#         epoch_loss = running_loss / trunning_corrects
#         epoch_acc = (running_corrects.double()*100) / trunning_corrects
#         print('\t\t Training: Epoch({}) - Loss: {:.4f}, Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
#
#         model.eval()
#         vrunning_loss = 0.0
#         vrunning_corrects = 0
#         num_samples = 0
#         for data, labels in valid_loader:
#             data = data.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#             with torch.no_grad():
#                 outputs = model(data)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)
#             vrunning_loss += loss.item() * data.size(0)
#             vrunning_corrects += (preds == labels).sum()
#             num_samples += preds.size(0)
#         vepoch_loss = vrunning_loss/num_samples
#         vepoch_acc = (vrunning_corrects.double() * 100)/num_samples
#         print('\t\t Validation({}) - Loss: {:.4f}, Acc: {:.4f}'.format(epoch, vepoch_loss, vepoch_acc))
