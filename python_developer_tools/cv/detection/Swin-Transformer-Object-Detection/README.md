## 项目路径
> https://github.com/SwinTransformer/Swin-Transformer-Object-Detection

## 安装成功的

```shell
conda create -n SwinTransformerObjectDetection -y python=3.8
conda activate SwinTransformerObjectDetection
conda install pytorch=1.8.0 torchvision cudatoolkit=10.2 -c pytorch -y

# mmcv安装
cd ~/software/
rm -rf mmcv
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .

# 安装apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# mmdetection安装
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -r requirements.txt
pip install -v -e . 

cd /home/zengxh/workspace/Swin-Transformer-Object-Detection
pip install -r requirements.txt

# 测试是否能用
cd /home/zengxh/workspace/Swin-Transformer-Object-Detection
python demo/image_demo.py demo/demo.jpg configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py mask_rcnn_swin_tiny_patch4_window7.pth
```

# 训练自己的数据集：
1. 准备coco数据集
2. 修改changemaskrcnn.py中num_class并运行
3. 修改configs\_base_\models\mask_rcnn_swin_fpn.py中num_classes
4. 修改configs\_base_\default_runtime.py中interval,load_from
5. 修改configs\swin\mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py中的max_epochs、lr
6. 修改configs\_base_\datasets\coco_instance.py中samples_per_gpu和workers_per_gpu
7. 修改mmdet\datasets\coco.py中CLASSES
8. python tools/train.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py
9. python demo/image_demo.py data/coco/val2017/2174638231500090-1_2_6.jpg configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco/latest.pth

## 异常
### 异常1：
> CocoDataset: Incompatible version of pycocotools is installed. Run pip uninstall pycocotools first. Then run pip install mmpycocotools to install open-mmlab forked pycocotools.
### 解决1：
```shell script
pip uninstall pycocotools
pip install mmpycocotools
pip install pycocotools
pip uninstall mmpycocotools
pip install pycocotools
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
pip uninstall pycocotools
pip install mmpycocotools
```


