#### 跳板机连接
```shell script
ssh -N -f -L 6001:10.123.33.2:22 -p 60022 xiaohui.zeng@10.121.1.60 -o TCPKeepAlive=yes
```