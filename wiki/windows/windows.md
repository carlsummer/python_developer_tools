#### 跳板机连接
```shell script
ssh -N -f -L 6001:10.123.33.2:22 -p 60022 xiaohui.zeng@10.121.1.60 -o TCPKeepAlive=yes
```
#### hosts
> C:\Windows\System32\drivers\etc

#### bat 判断命令是否执行成功
1. 连接符形式，&& 表示成功，|| 表示失败，例如：
```shell script
call xxx.bat && (goto succeed) || goto failed
:succeed
echo successfully
:failed
echo failed
pause
```
2. 使用%errorlevel%
```shell script
call xxx.bat
if %errorlevel% == 0 (
　　echo successfully
) else (
　　echo failed
)
```

### 杀死进程
```shell script
taskkill /pid 3548  -t  -f
```