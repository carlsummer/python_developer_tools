import json
import pylab as pl
import random
import numpy as np
import cv2

from python_developer_tools.cv.datasets.TT100K.anno_func import load_img,draw_all

datadir = r"D:\BaiduNetdiskDownload\TT100k\data"

filedir = datadir + "/annotations.json"
ids = open(datadir + "/train/ids.txt").read().splitlines()

annos = json.loads(open(filedir).read())

imgid = random.sample(ids, 1)[0]

imgdata = load_img(annos, datadir, imgid)
imgdata_draw = draw_all(annos, datadir, imgid, imgdata)
pl.figure(figsize=(20,20))
pl.imshow(imgdata_draw)
pl.savefig("2.jpg")

