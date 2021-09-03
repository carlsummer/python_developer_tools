
import argparse
import os
import re

from glob2 import glob
import cv2
import time
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
import numpy as np
from tqdm import tqdm
from imutils import paths
import shutil
from xml.dom.minidom import parse


def xmlFileObjNum(xml_file):
    big_dom_tree = parse(xml_file)
    big_root_node = big_dom_tree.documentElement
    objects = big_root_node.getElementsByTagName('object')
    if not len(objects):
        return True
    return False

def deleteEmptyXmlAndImg(small_root):
    xml_list = list(paths.list_files(small_root, '.xml'))
    for xml_file in tqdm(xml_list):
        if xmlFileObjNum(xml_file):
            os.remove(xml_file)
            os.remove(xml_file.replace('.xml', '.jpg'))

    # 如果图片没有对应xml，图片也删掉
    img_list = list(paths.list_images(small_root))
    for img_file in tqdm(img_list):
        if not os.path.exists(img_file.replace( '.jpg', '.xml')):
            os.remove(img_file)


if __name__ == '__main__':
    root = r'/home/zengxh/workspace/YOLOX/datasets/coco/org'
    deleteEmptyXmlAndImg(root)