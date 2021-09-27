# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/17/2021 4:11 PM
# @File:pandas_utils
import pandas as pd
import numpy as np
# 定义一个空的pd
kong=pd.DataFrame(columns=[i for i in range(30)])
# numpy 转 dataframe
dataDf=pd.DataFrame(np.arange(12).reshape(3,4))

# 将dataframe保存为csv文件
dataDf.to_csv("lgb.csv", index=False, encoding='utf_8_sig')

# 根据下标取值
dataDf.iloc[:, -20:]

# apply
dataDf.apply(lambda x: x.iloc[-20:] / np.max(x.iloc[-20:].values.flatten()) , axis=1)

# 删除某列
df = pd.DataFrame(np.arange(12).reshape(3,4), columns=['A', 'B', 'C', 'D'])
df.drop(['B', 'C'], axis=1)

# 过滤
"""
df = pd.read_csv('imdb.txt').sort(columns='year')
df[df['year']>1990].to_csv('filtered.csv')
alldataX.loc[(alldataX['calctype']==1) & (alldataX['iscoincidence'] ==1)]

pd.read_csv('imdb.txt')
  .sort(columns='year')
  .loc[lambda x: x['year']>1990]
  .to_csv('filtered.csv')
"""

# 多个csv合并
import pandas as pd
import os
filepath = 'D:\\PycharmProjects\\DataProcess\\Check_MatchResult\\'
outpath = 'C:\\Users\\user\\Desktop\\testout.csv'
allfile = os.listdir(filepath)
features = pd.DataFrame()
for file in allfile:
    feature = pd.read_csv(filepath + file, encoding='ANSI')
    features = features.append(feature)
features.to_csv(outpath, index=False)