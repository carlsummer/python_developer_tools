
import os
from tqdm import tqdm
import glob2

if __name__ == '__main__':
    rootpath = r'/home/zengxh/workspace/YOLOX/datasets/coco/org'
    dabiaotu_path = r"/home/zengxh/workspace/YOLOX/datasets/coco/org"
    xml_list = glob2.glob(rootpath + "/*.xml")
    for xml_path in tqdm(xml_list):
        img_path = xml_path.replace('.xml','.jpg')
        name = os.path.split(img_path)[1]
        path_file = os.path.join(dabiaotu_path, name)
        if not os.path.exists(path_file):
            print("xml:" + xml_path)
            os.remove(xml_path)
