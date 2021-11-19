git pull
echo "判断git pull命令是否执行正常"
if %errorlevel% == 0 (
	echo "successfully"
) else (
	echo "failed"
	pause
)
cd ..
echo "创建文件夹targz"
md targz
echo "拷贝ztpanels-dafeng"
XCOPY /e/y/r ztpanels-dafeng targz\ztpanels-dafeng\
echo "拷贝PVDefectAlgDeployV2DF"
XCOPY /e/y/r C:\Users\zengxh\Documents\workspace\git-chint-workspace\PVDefectAlgDeployV2DF targz\ztpanels-dafeng\extra_apps\PVDefectAlgDeployV2DF\
rd /s /q targz\ztpanels-dafeng\extra_apps\PVDefectAlgDeployV2DF\.git
cd ./targz
tar -czvf ztpanels-dafeng.tar.gz ztpanels-dafeng
echo "移动打包好的文件"
move ztpanels-dafeng.tar.gz ../
cd ..
rd /s /q targz
echo "打包完成"