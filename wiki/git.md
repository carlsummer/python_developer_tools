git clone ...
git log -1
git reset --hard log...

##### 有冲突时的操作
```shell script
git status
git add . # 将所有修改add到index
git stash save "2" # 将所有修改保存到管道
git pull # 拉取最新代码
git stash pop # 拉取刚刚保存在管道中的代码
# find >>>>> 进行修改该处
git add .
git commit -m "合并"
git push
```

##### 撤销add的某个文件
```shell script
git reset HEAD XXX/XXX/XXX.c 就是对某个文件进行撤销了
```
##### 查看所有分支c
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
##### 查看git项目在哪个路径下?
> 进入到命令行后,输入git remote -v

##### 查看git项目是从git的哪个分支上拉下来的命令?
> 如果还想看项目是从git的那个分支上拉下来的,可以在命令行中输入:git remote show origin

##### 回滚远程服务器
1. git reset --hard hash 回滚本地git库
2. git push -f origin（git仓库的url） branch名 强制提交

##### 将不同的代码保存到文件
```shell script
os.system(f"git diff HEAD > {outdir}/gitdiff.patch")
```