import sys

import argparse
import glob
import os
import shutil

from imutils import paths
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from python_developer_tools.cv.detection.yolovx.labelme2YoloXdatasets.txt2jsoncoco import labelme2coco
from python_developer_tools.files.common import resetDir
from python_developer_tools.python.threadings.multiprocessing_utils import parmap



def saveImg(xdata):
    shutil.copy(xdata["imgpath"], xdata["imagesDir"])


def xmlImg2coco(bigImgallPathList, bigXmlallPathList, datasetsDir):
    X_train, X_test_val, y_train, y_test_val = train_test_split(bigImgallPathList, bigXmlallPathList,
                                                        test_size=0.4,
                                                        random_state=1024)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=1024)

    annotationsDir = os.path.join(datasetsDir, "annotations")
    resetDir(annotationsDir)

    imagesDirG = os.path.join(datasetsDir, "train2017")
    resetDir(imagesDirG)
    X_train = [{"imgpath":imgpath,"imagesDir":imagesDirG} for imgpath in X_train]
    parmap(saveImg, X_train, 16)
    labelme2coco(y_train,os.path.join(annotationsDir, "instances_train2017.json"))

    imagesDirG = os.path.join(datasetsDir, "test2017")
    resetDir(imagesDirG)
    X_test = [{"imgpath":imgpath,"imagesDir":imagesDirG} for imgpath in X_test]
    parmap(saveImg, X_test, 16)
    labelme2coco(y_test, os.path.join(annotationsDir,"instances_test2017.json"))

    imagesDirG = os.path.join(datasetsDir, "val2017")
    resetDir(imagesDirG)
    X_val = [{"imgpath":imgpath,"imagesDir":imagesDirG} for imgpath in X_val]
    parmap(saveImg, X_val, 16)
    labelme2coco(y_val, os.path.join(annotationsDir, "instances_val2017.json"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将人工复判的img和xml 重新画图并保存到一个新的文件夹中")
    parser.add_argument('--sourceDir',
                        default=r'/home/zengxh/medias/data/ext/PVDefectData/haining/peiyu/9bb/danjing/banpian/166/daojiao/24/update_0825/',
                        help="复判好的文件夹")
    parser.add_argument('--cocodatasetDir',
                        default=r'/home/zengxh/workspace/YOLOX/datasets/coco/',
                        help="复判好的文件夹")
    args = parser.parse_args()
    sourceDir = args.sourceDir
    cocodatasetDir = args.cocodatasetDir
    # cocodatasetDir = os.path.abspath(os.path.join(sourceDir, "..")) if cocodatasetDir == "" else cocodatasetDir

    imglist = list(paths.list_images(sourceDir))
    xmlList = [imgpath.replace(".jpg", ".xml") for imgpath in imglist]
    xmlImg2coco(imglist, xmlList, cocodatasetDir)
