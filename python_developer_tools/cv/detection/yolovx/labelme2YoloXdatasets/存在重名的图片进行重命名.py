# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:10/11/2021 10:08 AM
# @File:存在重名的图片进行重命名
import argparse
import os

import shutil
import time

from imutils import paths

from python_developer_tools.files.common import get_filename_suf_pix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将人工复判的img和xml 重新画图并保存到一个新的文件夹中")
    parser.add_argument('--aDir',
                        default=r'/home/zengxh/workspace/YOLOX/datasets/coco/org/',
                        help="文件夹a")
    parser.add_argument('--bDir',
                        default=r'/home/zengxh/workspace/YOLOX/datasets/coco/origin/9-24-b/',
                        help="文件夹b")
    args = parser.parse_args()
    aDir = args.aDir
    bDir = args.bDir

    imglista = list(paths.list_images(aDir))
    imgnamea=[]
    for imga in imglista:
        filename, filedir, filesuffix, filenamestem = get_filename_suf_pix(imga)
        imgnamea.append(filename)
        
    imglistb = list(paths.list_images(bDir))
    for imgb in imglistb:
        filename, filedir, filesuffix, filenamestem = get_filename_suf_pix(imgb)
        if filename in imgnamea:
            newfilename = str(time.time()).replace(".","")
        else:
            newfilename = filenamestem
            
        shutil.copy(imgb, os.path.join(aDir, "{}{}".format(newfilename, filesuffix)))
        shutil.copy(imgb.replace(filesuffix, ".xml"), os.path.join(aDir, "{}{}".format(newfilename, ".xml")))

