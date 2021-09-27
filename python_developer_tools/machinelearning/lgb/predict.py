import pickle
import time

from sklearn.metrics import f1_score, mean_absolute_error

import lightgbm as lgb
import cv2
import numpy as np
import pandas as pd


if __name__ == '__main__':
    csvpath = r"datasets/datasets.csv"
    csvdf = pd.read_csv(csvpath, header=0)

    alldataX = csvdf[["calctype","col","row","convert_rate","iscoincidence","sobeld","jlxd","a1","a2","a3","a4","a5","a6","a7","a8",
                      "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12","s13","s14","s15","s16","s17","s18","s19","s20"]]
    alldataY = csvdf["Y"]

    # 模型加载
    # gbm = lgb.Booster(model_file=config.model_file[0])
    with open('lgb.pickle', 'rb') as f:
        gbm = pickle.load(f)

    # 模型预测
    start = time.time()
    y_pred = gbm.predict(alldataX[:1].values)
    print(time.time()-start)

    score = mean_absolute_error(y_pred, alldataY[:1])
    print(score)
