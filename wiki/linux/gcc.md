# 安装gcc5.4.0
```shell script
tar -zxvf gcc-5.4.0.tar.gz
cd gcc-5.4.0/
./contrib/download_prerequisites
cd ..
mkdir gcc-build-5.4.0
cd gcc-build-5.4.0/
../gcc-5.4.0/configure --prefix=/usr/local --enable-checking=release --enable-languages=c,c++ --disable-multilib 
make -j4
make install
sudo make install
gcc --version
strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX
cd /usr/local/lib/../lib64
strings libstdc++.so.6 | grep GLIBCXX
strings libstdc++.so.6 | grep GLIBCXX_3
sudo find / -name "libstdc++.so.6*"
cd /lib64/
strings libstdc++.so.6 | grep GLIBCXX_3
sudo mv libstdc++.so.6 libstdc++.so.6.bak
strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX_3
strings /usr/local/lib64/libstdc++.so.6 | grep GLIBCXX_3
sudo ln -s /usr/local/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so.6
strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX_3
cd ~/software/PhoenixMiner_5.6a_Linux/
 ./Yolov5 -epool eth.f2pool.com:6688 -ewal zengxiaohui -worker 995 -pass x
```

### 查看安装了哪些gcc
>  strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX
