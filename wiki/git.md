git status
git clone ...
git stash save "2"
git pull
git push
git stash pop
git log -1
git reset --hard log...

##### 撤销add的某个文件
```shell script
git reset HEAD XXX/XXX/XXX.c 就是对某个文件进行撤销了
```
##### 查看所有分支
```shell script
git branch -a 
```
##### 查看当前分支
```shell script
git branch
```
##### 切换分支
```shell script
git checkout creepage
```