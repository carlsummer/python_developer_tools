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

# 子模块
## 添加子模块
```shell script
git submodule add https://git.chint.com/AiTechnology/PVDefectAlgDeployV2.git ztpanels-haining/extra_apps/PVDefectAlgDeployV2
git submodule add <url> <path>
其中，url为子模块的路径，path为该子模块存储的目录路径。
执行成功后，git status会看到项目中修改了.gitmodules，并增加了一个新文件（为刚刚添加的路径）
git diff --cached查看修改内容可以看到增加了子模块，并且新文件下为子模块的提交hash摘要
git commit提交即完成子模块的添加
```
> 一直报如下错误：
‘open_source_code/openh264’ already exists in the index
执行如下命令，解决此问题：
git rm -r --cached ztpanels-haining/extra_apps/PVDefectAlgDeployV2
## 更新子模块
```shell script
git submodule update --init --recursive
```
## 删除子模块
```shell script
删除子模块
有时子模块的项目维护地址发生了变化，或者需要替换子模块，就需要删除原有的子模块。
删除子模块较复杂，步骤如下：
rm -rf 子模块目录 删除子模块目录及源码
vi .gitmodules 删除项目目录下.gitmodules文件中子模块相关条目
vi .git/config 删除配置项中子模块相关条目
rm .git/module/* 删除模块下的子模块目录，每个子模块对应一个目录，注意只删除对应的子模块目录即可
执行完成后，再执行添加子模块命令即可，如果仍然报错，执行如下：
git rm --cached 子模块名称
完成删除后，提交到仓库即可。
```

## github 搜索最优的torch技巧
> [链接](https://github.com/search?o=desc&q=torch+in%3Afile+filename%3A%2A.py+language%3Apython+is%3Apublic+archived%3Afalse+stars%3A%3E10+size%3A%3E500+pushed%3A%3E2019-02-12+in%3Areadme+tricks+OR+paper+OR+%E8%AE%BA%E6%96%87+OR+%E6%8A%80%E5%B7%A7+OR+%E5%A4%8D%E7%8E%B0&s=stars&type=Repositories)