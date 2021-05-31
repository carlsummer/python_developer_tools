#### 下载路径
> https://repo.anaconda.com/archive/

#### 创建虚拟环境
```shell script
conda create -n CenterNet2 python=3.8
# 创建虚拟环境名为：yolact_py38_yolov5 在 /home/admin/anaconda3/envs目录下
conda create --prefix=/home/admin/anaconda3/envs/yolact_py38_yolov5 python=3.8
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
#### 激活虚拟环境
```shell script
conda activate CenterNet2
```
#### 删除虚拟环境
```shell script
conda remove CenterNet2
```
#### 退出虚拟环境
```shell script
conda deactivate
```