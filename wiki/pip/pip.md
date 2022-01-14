##### pip安装包:
```shell script
# 搭建虚拟环境
virtualenv itbag
# pip 安装包
pip install labelImg -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
# pip 安装
pip install -r requirements.txt
# 更新pip
python.exe -m pip install --upgrade pip
```
##### pip install mysqlclient
> 如果报错失败了，那么
> sudo yum install mysql-devel

#### pip 缓存位置
> /home/admin/.cache/pip/wheels/

### window 设置pip源
> pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/