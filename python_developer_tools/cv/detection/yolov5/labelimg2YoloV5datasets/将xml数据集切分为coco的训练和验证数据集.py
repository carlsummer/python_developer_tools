import sys

import argparse
import glob
import os
import shutil

from sklearn.model_selection import train_test_split

from python_developer_tools.files.common import resetDir


def xmlImg2coco(bigImgallPathList, bigXmlallPathList, datasetsDir):
    X_train, X_test, y_train, y_test = train_test_split(bigImgallPathList, bigXmlallPathList,
                                                        test_size=0.2,
                                                        random_state=1024)

    annotationsTrainDir = os.path.join(datasetsDir, "annotations/train2017")
    imagesTrainDir = os.path.join(datasetsDir, "images/train2017")
    annotationsValDir = os.path.join(datasetsDir, "annotations/val2017")
    imagesValDir = os.path.join(datasetsDir, "images/val2017")

    print(imagesTrainDir)
    resetDir(imagesTrainDir)
    resetDir(annotationsTrainDir)
    for xtrain, ytrain in zip(X_train, y_train):
        shutil.copy(xtrain, imagesTrainDir)
        shutil.copy(ytrain, annotationsTrainDir)

    resetDir(imagesValDir)
    resetDir(annotationsValDir)
    for xtest, ytest in zip(X_test, y_test):
        shutil.copy(xtest, imagesValDir)
        shutil.copy(ytest, annotationsValDir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将人工复判的img和xml 重新画图并保存到一个新的文件夹中")
    parser.add_argument('--sourceDir',
                        default=r'/home/zengxh/workspace/YOLOX/datasets/coco/org/',
                        help="复判好的文件夹")
    parser.add_argument('--cocodatasetDir',
                        default=r'/home/zengxh/workspace/YOLOX/datasets/coco/',
                        help="复判好的文件夹")
    args = parser.parse_args()
    sourceDir = args.sourceDir
    cocodatasetDir = args.cocodatasetDir
    # cocodatasetDir = os.path.abspath(os.path.join(sourceDir, "..")) if cocodatasetDir == "" else cocodatasetDir

    imglist = glob.glob(os.path.join(sourceDir, '*.jpg'))
    xmlList = [imgpath.replace(".jpg", ".xml") for imgpath in imglist]
    xmlImg2coco(imglist, xmlList, cocodatasetDir)
