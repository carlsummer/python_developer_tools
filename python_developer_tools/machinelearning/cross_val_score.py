# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/19/2021 5:04 PM
# @File:cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

X, y = datasets.load_iris(return_X_y=True)

clf_svc_cv = svm.SVC(kernel='linear',C=1)
scores_clf_svc_cv = cross_val_score(clf_svc_cv,X,y,cv=5)
print(scores_clf_svc_cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_clf_svc_cv.mean(), scores_clf_svc_cv.std() * 2))