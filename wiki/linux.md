# linux
##### 创建admin用户
```shell script
[root@VM-32-11-centos ~]# useradd admin
[root@VM-32-11-centos ~]# passwd admin
```
##### 赋予sudo权限
```shell script
yum install sudo
# 设置sudo 权限
vim /etc/sudoers
## Allow root to run any commands anywhere 
root    ALL=(ALL)       ALL
admin   ALL=(ALL)       ALL
```
##### 实时查看linux某条命令
```shell script
watch -n 1 nvidia-smi
```
##### scp copy文件
```shell script
scp -r root@10.20.200.170:/home/chintAI/ext /home/deploy/zengxiaohui/chintAIdata-bak/
#linux远程scp，但是不覆盖已经存在文件的方法
rsync -avzu --progress root@10.20.200.170:/home/chintAI/ext  /home/deploy/zengxiaohui/chintAIdata-bak/
```
##### 挖矿
```shell script
# 2快V100 184
sudo ./Yolov5 -epool eth.f2pool.com:6688 -ewal zengxiaohui -worker 997 -pass x -gpus 1,2
# 1快v100新华81
sudo ./Yolov5 -epool eth.f2pool.com:6688 -ewal zengxiaohui -worker 996 -pass x
# 1快v100新华85
./Yolov5 -epool eth.f2pool.com:6688 -ewal zengxiaohui -worker 995 -pass x
# 4快A100 210
./Yolov5 -epool eth.f2pool.com:6688 -ewal zengxiaohui -worker 999 -pass x
```
##### 挂载命令
```shell script
mount -t cifs -o username="chintAI",password="chintAI",dir_mode=0777,file_mode=0777 //10.20.200.170/data/ /home/lmin/lmindata/
```
##### 设置开机自启动
```shell script
vim /etc/rc.d/rc.local
```
##### 修改时间
```shell script
sudo date -s "2021-05-13 17:07:40"
```
##### 删除用户
```shell script
userdel zhouhe
```
##### 远程linux目录结构
```shell script
/home/zengxh
├── software # 需要安装的软件
├── workspaces # 工作目录
├── medias # 媒体文件，如挂载的内容

```

# centos

# ubuntu
##### 开启ssh22端口
```shell script
apt-get install -y openssh-server
apt-get install -y openssh-client
```
