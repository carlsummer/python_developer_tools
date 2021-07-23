#### 下载路径
> https://repo.anaconda.com/archive/

#### 创建虚拟环境
```shell script
# 在线安装
conda create -n yolact_py38_yolov5 python=3.8 
# 创建虚拟环境名为：yolact_py38_yolov5 在 /home/admin/anaconda3/envs目录下
conda create --prefix=/home/admin/anaconda3/envs/yolact_py38_yolov5 python=3.8
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
#离线安装
conda create -n yolact_py38_yolov5 --offline python=3.8

虚拟环境名称：your_project
首先/xxx/anaconda3/envs/your_project
conda create -n new_project --clone ./your_project --offline
注意：/xxx/anaconda3/pkgs 也要一同拷贝到新机器上的对应位置/new_xxx/anaconda3/pkgs
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