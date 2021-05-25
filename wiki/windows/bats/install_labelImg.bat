echo "开始安装labelImg"
pip install labelImg -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
echo "安装完成"

echo "获取labelimg的路径"
for /F %%i in ('where labelImg') do ( set commitid=%%i)
echo commitid=%commitid%

echo "获取桌面路径"
set d=%USERPROFILE%\Desktop\


echo "开始创建labelImg快捷方式"
mshta VBScript:Execute("Set a=CreateObject(""WScript.Shell""):Set b=a.CreateShortcut(a.SpecialFolders(""Desktop"") & ""\labelImg.lnk""):b.TargetPath=""%commitid%"":b.WorkingDirectory=""%d%"":b.Save:close")
echo "创建成功"

labelImg
pause