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
├── datasets # 数据集
```

##### 统计当前目录下，排除venv目录，剩余所有py文件的行数
```shell script
wc -l `find  -path ./venv -prune -o -name '*py'`
```
##### linux下查看进程运行的时间
```shell script
ps -eo pid,tty,user,comm,lstart,etime | grep 28156
> 28156 pts/4    zhouhe   Enet            Sat May 22 06:30:15 2021  2-15:53:30
> pid：28156
> tty：pts/4
> user：zhouhe
> comm：Enet
> lstart： Sat May 22 06:30:15 2021 【开始时间为：2021-5-22  06:30:15 周六】
> etime：2-15:53:30 【运行时间：2天15个小时53分钟30秒】
```
##### 修改文件夹所属用户和组
```shell script
# 将文件夹/home/deploy/datasets/coco 修改组为：deploy
chgrp -R deploy /home/deploy/datasets/coco
# 修改文件夹以及其子目录的文件使用-R选项 用户
chown -R deploy /home/deploy/datasets/coco
```
##### ssh连接
```shell script
ssh -p 6002 admin@127.0.0.1 
```
##### 替换sh文件中的\r
```shell script
sed -i 's/\r//' run.sh
```
##### 将文件夹打包为gz文件
```shell script
cd C:\Users\zengxh\Documents\workspace\git-chint-workspace\PVDefectPlatform
tar -czvf ztpanels-haining.tar.gz ztpanels-haining
```
##### 查看内存大小
```shell script
free -h
```
##### 禁止访问百度
```shell script
iptables -A OUTPUT -p tcp -d www.baidu.com --dport 80 -j DROP
iptables -L //生效
 ```
##### 允许访问百度
```shell script
iptables -A OUTPUT -p tcp -d www.baidu.com --dport 80 -j ACCEPT
iptables -L
```
##### 查看最近1天home目录下修改的python文件
```shell script
sudo find /home/ -name '*.py' -ctime 0 -ls
```
# centos
##### ftp 服务器搭建
```shell script
https://blog.csdn.net/qq_36938617/article/details/89077845
yum install vsftpd ftp
配置文件在/etc/vsftpd/vsftpd.conf
我是把匿名用户的权限给guanle
我把10.123.33.2上的selinux修改成了Permissive
eg:wget ftp://10.123.33.2/workspaces/ztpanels-haining.tar.gz --ftp-user=admin --ftp-password=Ztadmin2020
--ftp-user 是ftp用户名
--ftp-password 是用户密码
```

# ubuntu
##### 开启ssh22端口
```shell script
apt-get install -y openssh-server
apt-get install -y openssh-client
```
