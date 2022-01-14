# linux
##### 创建admin用户
```shell script
[root@VM-32-11-centos ~]# useradd admin
[root@VM-32-11-centos ~]# passwd admin
# 删除用户
[root@VM-32-11-centos ~]# userdel -r shenjh    
```
##### 赋予sudo权限
```shell script
yum install sudo
# 设置sudo 权限
vim /etc/sudoers
## Allow root to run any commands anywhere 
root    ALL=(ALL)       ALL
admin   ALL=(ALL)       ALL

# 方法二
# 将deploy添加到root组（wheel）
# 命令：
usermod -g root deploy   # -u root
# 之后deploy就拥有root的权限了

查看Linux某用户属于哪个组
id  deploy
groups deploy
```

##### 修改history
```shell script
修改文件~/.bash_history
```
#### 记录用户最后一次登录记录
```shell script
cd /var/log/
sudo rm -rf lastlog
sudo vim lastlog
```
```shell script
sudo rm -rf /var/run/utmp
sudo vim /var/run/utmp
who
sudo rm -rf /var/log/wtmp
sudo vim /var/log/wtmp
last 
sudo rm -rf /var/log/btmp
sudo vim /var/log/btmp
lastb
sudo rm -rf /var/log/lastlog
sudo vim /var/log/lastlog
lastlog
其中 utmp 对应w 和 who命令； wtmp 对应last命令；btmp对应lastb命令；lastlog 对应lastlog命令
```
##### 查看最近用户是在那台ip上登录的
```shell script
lastlog
```

##### 实时查看linux某条命令
```shell script
watch -n 1 nvidia-smi
```
##### scp copy文件
```shell script
scp  -P 22 -r root@10.20.200.170:/home/chintAI/ext /home/deploy/zengxiaohui/chintAIdata-bak/
#linux远程scp，但是不覆盖已经存在文件的方法
rsync -avzu --progress root@10.20.200.170:/home/chintAI/ext  /home/deploy/zengxiaohui/chintAIdata-bak/
```
##### 挖矿
```shell script
PhoenixMiner_5.6a_Linux
# 2快V100 184
sudo ./Yolov5 -epool eth.f2pool.com:6688 -ewal zengxiaohui -worker 997 -pass x -gpus 1,2
# 1快v100新华81
sudo ./Yolov5 -epool eth.f2pool.com:6688 -ewal zengxiaohui -worker 996 -pass x
# 1快v100新华82
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
##### 将/home/data 这个目录下的所有文件打包压缩为当前目录下的data.zip
```shell script
zip -q -r data.zip /home/data
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
##### xshell上传文件，下载文件
```shell script
rz
sz work.txt
```
##### 查看当前目录下python3的文件大小
```shell script
du -sh python3
```
##### 查看CPU信息
```shell script
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c
free -h # 查看内存
sudo dmidecode |grep -A16 "System Information$" # 查看主板型号
```



##### 查看端口占用情况
```shell script
netstat -anp|grep 80
```
#### 查看显卡数量和信息
```shell script
nvidia-smi -L
```

#### 统计文件夹下文件个数，包括子文件
```shell script
ls -lR | grep "^-"| wc -l
```
#### 统计文件夹下目录个数，包括子目录
```shell script
ls -lR | grep "^d"| wc -l
```
```shell
# 按时间顺序排序，只查看前10条信息
ls -ltr | head -10
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

wget -nH -m ftp://10.123.33.2/.cache/pip/ --ftp-user=admin --ftp-password=Ztadmin2020
```
#### rpm 安装
```shell script
# 安装 example.rpm 包并在安装过程中显示正在安装的文件信息及安装进度；
rpm -ivh example.rpm 
# 卸载 tomcat4 软件包
rpm -e tomcat4 
```
#### 强制移动覆盖
```shell script
mv -f A B
```

#### 开启8001端口
> 首先centos7的防火墙由iptables改为了firewalld
> 1. 执行命令：firewall-cmd --zone=public --add-port=80/tcp  --permanent
>     命令含义：
>     --zone #作用域
>     --add-port=80/tcp   #添加端口  格式为：端口/协议
>    --parmanent  #永久生效  没有此参数重启后失效
> 2. 重启防火墙：systemctl restart firewalld
```shell script
[root@localhost ~]# firewall-cmd --zone=public --add-port=8001/tcp  --permanent
success
[root@localhost ~]# systemctl restart firewalld
```

# ubuntu
##### 开启ssh22端口
```shell script
apt-get install -y openssh-server
apt-get install -y openssh-client
```

