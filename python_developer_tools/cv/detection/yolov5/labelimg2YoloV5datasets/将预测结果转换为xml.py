# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/2/2021 11:28 AM
# @File:将预测结果转换为xml
import json
import os
from pathlib import Path
from lxml.etree import Element, SubElement, tostring

def json_2_xml_save(save_xml_path, jsons, origin_imgname):
    root_path = Path(save_xml_path).parent
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'images'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = origin_imgname

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(0)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(0)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for box in jsons:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = box['tag']
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = "0"
        node_bndbox = SubElement(node_object, 'bndbox')

        coord_list = list(map(lambda x: int(float(x)), box['coordinate'].split(";")[0].split(',')[:4]))

        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(coord_list[0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(coord_list[1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(coord_list[2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(coord_list[3])

        node_row = SubElement(node_bndbox, 'row')
        node_row.text = str(box['row_idx'])
        node_col = SubElement(node_bndbox, 'col')
        node_col.text = str(box['col_idx'])

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    with open(os.path.join(root_path, origin_imgname.replace('.jpg', '.xml')), 'wb') as f:
        f.write(xml)
