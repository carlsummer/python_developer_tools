# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.10-py3

MAINTAINER zengxh <zengxh@chint.com>

# 修改apt-get源为国内源
ADD sources.list /etc/apt/

# Install linux packages
RUN apt update && apt install -y screen

# Install python dependencies
RUN pip install --upgrade pip

# 配置pip 国内源
RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com

# Create working directory
RUN mkdir -p /usr/src/ztpanels
WORKDIR /usr/src/ztpanels

# Copy contents
COPY . /usr/src/ztpanels

# 安装mysql相关软件
RUN apt-get -y install libmysqlclient-dev python3-dev

# 安装pip包
RUN pip install -r requirements.txt

# copy six文件
COPY six.py /opt/conda/lib/python3.6/site-packages/django/utils

# 设置环境变量
ENV TZ Asia/Shanghai
# 打开容器的8000端口
EXPOSE 8000

CMD [ "python","/usr/src/ztpanels/manage.py","runserver","0.0.0.0:8000","--noreload" ]