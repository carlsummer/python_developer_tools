# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:10/11/2021 9:47 AM
# @File:删除没有标准的图片
import argparse
import os

from imutils import paths

from python_developer_tools.files.common import get_filename_suf_pix

def delnotexists(imglist,filesuffix2=".xml"):
    for imgpath in imglist:
        filename, filedir, filesuffix, filenamestem = get_filename_suf_pix(imgpath)
        xmlpath = imgpath.replace(filesuffix,filesuffix2)
        if not os.path.exists(xmlpath) or not os.path.exists(imgpath):
            if os.path.exists(xmlpath):
                os.remove(xmlpath)
            if os.path.exists(imgpath):
                os.remove(imgpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将人工复判的img和xml 重新画图并保存到一个新的文件夹中")
    parser.add_argument('--sourceDir',
                        default=r'/home/zengxh/workspace/YOLOX/datasets/coco/origin/9-24-a/',
                        help="复判好的文件夹")
    args = parser.parse_args()
    sourceDir = args.sourceDir

    imglist = list(paths.list_images(sourceDir))
    delnotexists(imglist,".xml")
    imglist = list(paths.list_files(sourceDir,".xml"))
    delnotexists(imglist, ".bmp")
