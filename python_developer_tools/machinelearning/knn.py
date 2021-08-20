# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/20/2021 8:39 AM
# @File:knn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

kNN_classifier = KNeighborsClassifier(n_neighbors=6)
kNN_classifier.fit(X_train,y_train)
kNN_classifier.predict(X_test)
best_score = 0.0
best_k = -1
for k in range(1,11):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train,y_train)
    score = knn_clf.score(X_test,y_test)
    if score > best_score:
        best_k = k
        best_score=score
print("best_k=",best_k)
print("best_score=",best_score)
