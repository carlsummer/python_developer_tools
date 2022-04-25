import argparse
import copy
import os
import random
import shutil
from imutils import paths
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from xml.dom.minidom import parse
from lxml.etree import Element, SubElement, tostring

import xml.etree.ElementTree as ET
from python_developer_tools.cv.metrics.common import computeIOU
from python_developer_tools.files.xml_utils import read_xml
import json


def create_dir(defect_save_dir):
    if not os.path.exists(defect_save_dir):
        os.makedirs(defect_save_dir)

def not_super_sile(value1, maxval):
    if value1 <= 0:
        return 0
    if value1 >= maxval:
        return int(maxval)
    return int(value1)

def json_2_xml_save(save_xml_path, jsons, origin_imgname,image_height, image_width):
    # 将jsons保存为xml
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'images'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = origin_imgname

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(image_width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(image_height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for box in jsons:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = box['category']
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = "0"
        node_bndbox = SubElement(node_object, 'bndbox')

        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(not_super_sile(box['bbox']["xmin"],image_width))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(not_super_sile(box['bbox']["ymin"],image_height))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(not_super_sile(box['bbox']["xmax"],image_width))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(not_super_sile(box['bbox']["ymax"],image_height))

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    with open(save_xml_path, 'wb') as f:
        f.write(xml)



def clip_object_to_small(imgid):
    anno = annos["imgs"][imgid]
    imgpath = os.path.join(opt.source_dir, anno['path'])
    imgname = Path(anno['path']).name
    image = cv2.imread(imgpath)
    image_height, image_width, _ = image.shape

    json_2_xml_save(os.path.join(opt.save_dir,Path(anno['path']).stem + ".xml"),anno["objects"],imgname,image_height, image_width)
    cv2.imwrite(os.path.join(opt.save_dir, imgname),image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=str, default=r'D:\BaiduNetdiskDownload\TT100k\data',
                        help='需要处理的文件夹')
    parser.add_argument('--save-dir', type=str, default=r'D:\BaiduNetdiskDownload\tt100kxml\org',
                        help='提出来存放的位置')
    opt = parser.parse_args()

    filedir = opt.source_dir + "/annotations.json"
    ids = open(opt.source_dir + "/train/ids.txt").read().splitlines()
    ids2 = open(opt.source_dir + "/test/ids.txt").read().splitlines()
    ids.extend(ids2)
    annos = json.loads(open(filedir).read())

    create_dir(opt.save_dir)

    for id in tqdm(ids):
        clip_object_to_small(id)
