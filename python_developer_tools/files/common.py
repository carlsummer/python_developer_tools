# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/29/2021 4:22 PM
# @File:common
import os
import shutil
from pathlib import Path
import glob

def resetDir(dirpath):
    """判断文件夹是否存在存在那么删除重建，不存在那么创建"""
    if mkdir(dirpath):
        shutil.rmtree(dirpath)
        os.makedirs(dirpath)

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        return False
    else:
        return True

def get_filename_suf_pix(filepath):
    """
    :param filepath: '/home/deploy/datasets/creepageDistance/lr/test/0.jpg'
    :return: 0.jpg /home/deploy/datasets/creepageDistance/lr/test .jpg 0
    """
    path_obj = Path(filepath)
    filename, filedir, filesuffix, filenamestem = path_obj.name,str(path_obj.parent),path_obj.suffix,path_obj.stem
    return filename, filedir, filesuffix, filenamestem



def get_filelist(dir, Filelist):
    """获取文件夹及子文件夹下所有文件"""
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            # if s == "xxx":
            # continue
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist

def increment_dir(dir, comment=''):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    d = sorted(glob.glob(dir + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir):x.find('_') if '_' in x else None]) for x in d]) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')