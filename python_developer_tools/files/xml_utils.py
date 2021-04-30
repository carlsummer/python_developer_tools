# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/29/2021 4:22 PM
# @File:xml_utils
from xml.dom.minidom import parse
from lxml.etree import Element, SubElement, tostring


def read_predict_xml(label_path):
    # 读取xml中的内容
    gt_lists = []
    # 获取所有xml中object对象
    small_dom_tree = parse(label_path)
    small_root_node = small_dom_tree.documentElement
    objects = small_root_node.getElementsByTagName('object')

    for obj in objects:
        name = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
        if name in ["cell", "barcode"]:
            continue
        xmin = obj.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
        ymin = obj.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
        xmax = obj.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
        ymax = obj.getElementsByTagName('ymax')[0].childNodes[0].nodeValue
        gt = {"coordinate": "{},{},{},{}".format(xmin, ymin, xmax, ymax),
              "col_idx": "",
              "row_idx": "",
              "prob": "",
              "tag": name}
        gt_lists.append(gt)
        #
        #     pt_x1, pt_y1, pt_x2, pt_y2 = pt["coordinate"].split(",")
        #     pt_col, pt_row = pt["col_idx"], pt["row_idx"]
        #     pt_label = pt["tag"]
    return gt_lists


def json_2_xml_save(save_xml_path, jsons, origin_imgname):
    # 将jsons保存为xml
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

        coord_list = list(map(lambda x: int(float(x)), box['coordinate'].split(',')))

        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(coord_list[0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(coord_list[1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(coord_list[2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(coord_list[3])

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    with open(save_xml_path, 'wb') as f:
        f.write(xml)
