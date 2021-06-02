### 项目路径
> https://github.com/d-li14/involution.git

### 环境配置
```shell script
conda create -n involution -y python=3.8
conda activate involution
conda install pytorch=1.8.0 torchvision cudatoolkit=10.2 -c pytorch -y

# mmcv安装
cd ~/software/involution/
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .

# mmdetection安装
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements.txt
python setup.py develop

# 测试是否安装完成
python demo/image_demo.py demo/demo.jpg configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

# 安装cocoapi
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

### 准备训练自己的数据集
```shell script
修改mmdet\datasets\coco.py的 CLASSES  

准备coco数据集
```

#### 修改faster_runn
```shell script
获取involution 的代码
git clone https://github.com/d-li14/involution.git
将det的里面的代码copy到对应的目录中

1. 修改configs\_base_\models\faster_rcnn_red50_neck_fpn.py的num_classes
2. 修改configs\_base_\datasets\coco_detection.py的samples_per_gpu、workers_per_gpu
3. 修改configs\_base_\schedules\schedule_1x_warmup.py的lr、total_epochs
```