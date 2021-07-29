# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/26/2021 9:57 AM
# @File:optimizer

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1

optimizer = optim.Adam(model.parameters(), lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
