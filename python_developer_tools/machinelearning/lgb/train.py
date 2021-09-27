import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import svm
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold, ShuffleSplit
import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def train_test_split_fun(alldataX, alldataY):
    return train_test_split(alldataX, alldataY, test_size=0.2, random_state=1024)


if __name__ == '__main__':
    csvpath = r"datasets/datasets.csv"
    csvdf = pd.read_csv(csvpath, header=0)

    alldataX = csvdf[
        ["calctype", "col", "row", "convert_rate", "iscoincidence", "sobeld", "jlxd", "a1", "a2", "a3", "a4", "a5",
         "a6", "a7", "a8",
         "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17",
         "s18", "s19", "s20"]]
    alldataY = csvdf["Y"]
    # 0.04288846720587341
    # 0.04579131922898897
    # alldataX.iloc[:, -20:] = alldataX.apply(lambda x: x.iloc[-20:] / np.max(x.iloc[-20:].values), axis=1)
    # alldataX["col"] =alldataX.apply(lambda x: x["col"] /24, axis=1)
    # alldataX["row"] =alldataX.apply(lambda x: x["row"] /6, axis=1)

    train_X, test_X, train_Y, test_Y = train_test_split_fun(alldataX.values, alldataY.values)

    # build_model()
    clf = lgb.LGBMRegressor()
    # clf.fit(train_X, train_Y,eval_set=[(test_X, test_Y)])
    clf.fit(alldataX.values, alldataY.values)
    with open('lgb.pickle', 'wb') as f:
        pickle.dump(clf, f)

    train_predict = clf.predict(train_X)
    score = mean_absolute_error(train_predict, train_Y)
    print(score)

    test_predict = clf.predict(test_X)
    score = mean_absolute_error(test_predict, test_Y)
    print(score)
