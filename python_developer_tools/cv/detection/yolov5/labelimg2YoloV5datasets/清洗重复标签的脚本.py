# !/usr/bin/env python

'''
这个脚本是删除标签重复的，比如一个位置，有三个重复标签
'''


# -*- coding: utf-8 -*-

import os
from glob2 import glob
from xml.dom.minidom import parse
import shutil
from imutils import paths
from tqdm import tqdm
import xml
from xml.dom import minidom
import codecs
import cv2
import numpy as np
# import torch
from lxml.etree import Element, SubElement, tostring
import re

# 使用说明

def fixed_writexml(self, writer, indent="", addindent="", newl=""):
    writer.write(indent + "<" + self.tagName)

    attrs = self._get_attributes()
    a_names = attrs.keys()
    # a_names.sort()

    for a_name in a_names:
        writer.write(" %s=\"" % a_name)
        xml.dom.minidom._write_data(writer, attrs[a_name].value)
        writer.write("\"")
    if self.childNodes:
        if len(self.childNodes) == 1 and self.childNodes[0].nodeType == xml.dom.minidom.Node.TEXT_NODE:
            writer.write(">")
            self.childNodes[0].writexml(writer, "", "", "")
            writer.write("</%s>%s" % (self.tagName, newl))
            return
        writer.write(">%s" % (newl))
        for node in self.childNodes:
            if node.nodeType is not xml.dom.minidom.Node.TEXT_NODE:
                node.writexml(writer, indent + addindent, addindent, newl)
        writer.write("%s</%s>%s" % (indent, self.tagName, newl))
    else:
        writer.write("/>%s" % (newl))


xml.dom.minidom.Element.writexml = fixed_writexml


def produceBigXMl(big_xml_path, loc):

    for loc_ in loc:
        if '.jpg' in loc_:
            file_name_w_h_d = loc_.split('#')

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'PVD'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = file_name_w_h_d[0]

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = file_name_w_h_d[1]

    node_height = SubElement(node_size, 'height')
    node_height.text = file_name_w_h_d[2]

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = file_name_w_h_d[3]

    for offsets in loc:
        if '.jpg' not in offsets:
            offset_info = offsets.split('#')

            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = offset_info[0]
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_location = SubElement(node_object, 'location')
            node_location.text = offset_info[1]
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(offset_info[2])
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(offset_info[3])
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(offset_info[4])
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(offset_info[5])

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行

    with open(big_xml_path, 'wb') as f:
        f.write(xml)


def compareIou(yiwu_val, next_val):
    xmin1, ymin1, xmax1, ymax1 = [round(float(val)) for val in yiwu_val]
    xmin2, ymin2, xmax2, ymax2 = [round(float(val)) for val in next_val]

    # 求交集部分左上角的点
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    # 求交集部分右下角的点
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    if xmin > xmax or ymin > ymax:
        return False
    # 计算输入的两个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    # 计算总面积
    s = s1 + s2
    # 计算交集
    inter_area = (xmax - xmin) * (ymax - ymin)
    if (s - inter_area) <= 0:
        return  False
    iou = inter_area / (s - inter_area)
    return True if round(iou, 2) >= 0.85 else False

def delete_more_biaoqian():
    file_dir = r'/home/zengxh/workspace/YOLOX/datasets/coco/org'
    xml_list = list(paths.list_files(file_dir, '.xml'))
    for xml_file in tqdm(xml_list):

        xml_name = os.path.split(xml_file)[1]

        loc_list = []

        big_dom_tree = parse(xml_file)
        big_root_node = big_dom_tree.documentElement
        filename = big_root_node.getElementsByTagName('filename')[0].childNodes[0].nodeValue
        img_width = size = big_root_node.getElementsByTagName('width')[0].childNodes[0].nodeValue
        img_height = size = big_root_node.getElementsByTagName('height')[0].childNodes[0].nodeValue
        depth = size = big_root_node.getElementsByTagName('depth')[0].childNodes[0].nodeValue
        file_name_str = '#'.join([filename, img_width, img_height, depth])

        loc_list.append(file_name_str)

        objects = big_root_node.getElementsByTagName('object')
        for obj in objects:
            name = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
            # location = obj.getElementsByTagName('location')[0].childNodes[0].nodeValue
            find_last = re.findall('\d+_\d+.xml', xml_name)
            location = find_last[0].replace('.xml', '')
            xmin = str(obj.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = str(obj.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = str(obj.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = str(obj.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            loc_str = '#'.join([name, location, xmin, ymin, xmax, ymax])
            loc_list.append(loc_str)

        loc_list = list(set(loc_list))

        produceBigXMl(xml_file, loc_list)
        delete_big_than(xml_file)




# 删除 IOU > 0.8 的框
def delete_big_than(xml_file):
    xml_name = os.path.split(xml_file)[1]

    loc_list = []

    big_dom_tree = parse(xml_file)
    big_root_node = big_dom_tree.documentElement
    filename = big_root_node.getElementsByTagName('filename')[0].childNodes[0].nodeValue
    img_width = size = big_root_node.getElementsByTagName('width')[0].childNodes[0].nodeValue
    img_height = size = big_root_node.getElementsByTagName('height')[0].childNodes[0].nodeValue
    depth = size = big_root_node.getElementsByTagName('depth')[0].childNodes[0].nodeValue
    file_name_str = '#'.join([filename, img_width, img_height, depth])

    loc_list.append(file_name_str)

    objects = big_root_node.getElementsByTagName('object')
    for obj in objects:
        name = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
        # location = obj.getElementsByTagName('location')[0].childNodes[0].nodeValue
        find_last = re.findall('\d+_\d+.xml', xml_name)
        location = find_last[0].replace('.xml', '')
        xmin = str(obj.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
        ymin = str(obj.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
        xmax = str(obj.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
        ymax = str(obj.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
        loc_str = '#'.join([name, location, xmin, ymin, xmax, ymax])
        loc_list.append(loc_str)

    loc_list = list(set(loc_list))

    need_delte = []
    jpg = []
    not_jpg_list = []
    for val in loc_list:
        if '.jpg' not in val:
            not_jpg_list.append(val)
        else:
            jpg.append(val)

    for pre in range(len(not_jpg_list) - 1):
        for next_ in range(pre + 1, len(not_jpg_list)):
            if compareIou(not_jpg_list[pre].split('#')[2:], not_jpg_list[next_].split('#')[2:]):
                need_delte.append(not_jpg_list[next_])

    not_jpg_list.append(jpg[0])
    need_delte = list(set(need_delte))
    for delete_loc in need_delte:
        not_jpg_list.remove(delete_loc)

    produceBigXMl(xml_file, not_jpg_list)
    # print(not_jpg_list)

if __name__ == '__main__':
    delete_more_biaoqian()