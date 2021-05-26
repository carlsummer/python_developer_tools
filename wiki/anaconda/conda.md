#### 下载路径
> https://repo.anaconda.com/archive/

#### 创建虚拟环境
```shell script
conda create -n CenterNet2 python=3.8
conda activate CenterNet2
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

conda deactivate
conda remove CenterNet2
```