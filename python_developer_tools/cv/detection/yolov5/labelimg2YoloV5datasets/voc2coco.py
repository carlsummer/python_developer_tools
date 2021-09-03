import sys
import argparse
import copy
import os
from glob import glob
import json
import cv2
import re
from tqdm import tqdm

pattens = ['name', 'xmin', 'ymin', 'xmax', 'ymax']


def get_annotations(xml_path):
    bbox = []
    with open(xml_path, 'r') as f:
        text = f.read().replace('\n', 'return')
        p1 = re.compile(r'(?<=<object>)(.*?)(?=</object>)')
        result = p1.findall(text)
        for obj in result:
            tmp = []
            for patten in pattens:
                p = re.compile(r'(?<=<{}>)(.*?)(?=</{}>)'.format(patten, patten))
                if patten == 'name':
                    tmp.append(p.findall(obj)[0])
                else:
                    tmp.append(int(float(p.findall(obj)[0])))
            bbox.append(tmp)
    return bbox


def xml2cocoByDir(dirpath="train2017"):
    jpg_path_list = glob(os.path.join(sourceDir, "images/{}".format(dirpath), "*.jpg"))

    for imgpath in tqdm(jpg_path_list):
        xmlpath = imgpath.replace('.jpg', '.xml')
        xmlpath = xmlpath.replace('images', 'annotations')
        image = cv2.imread(imgpath)
        gheight, gwidth, _ = image.shape
        bbox = get_annotations(xmlpath)

        labelsdirpath = os.path.join(sourceDir, 'labels/{}/'.format(dirpath))
        if not os.path.exists(labelsdirpath):
            os.makedirs(labelsdirpath)

        [fpath, fname] = os.path.split(imgpath)
        fname1 = fname.split('.')[0]
        with open(os.path.join(labelsdirpath, fname1 + '.txt'), 'a') as seg_txt:
            for bb in bbox:
                #/home/admin/PVDefect/DataAnnotationPlatform/common/yolov5-labels.json
                yolov5Labels = json.load(open("yolov5-labels.json", "r", encoding="UTF-8"))
                cls = str(yolov5Labels[bb[0].strip()]["yolov5LabelNum"])
                x_center = str((bb[1] + bb[3]) * 0.5 / gwidth)
                y_center = str((bb[2] + bb[4]) * 0.5 / gheight)
                width = str((bb[3] - bb[1]) * 1.0 / gwidth)
                height = str((bb[4] - bb[2]) * 1.0 / gheight)
                seg_txt.write(cls + ' ' + x_center + ' ' + y_center + ' ' + width + ' ' + height + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="voc xml 转txt coco数据集格式")
    parser.add_argument('--sourceDir', default=r"/home/zengxh/workspace/YOLOX/datasets/coco/",
                        help="需要转换的txt文件夹")
    args = parser.parse_args()
    sourceDir = args.sourceDir

    xml2cocoByDir("train2017")
    xml2cocoByDir("val2017")
