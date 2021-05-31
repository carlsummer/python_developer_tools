cd /home/admin/PVDefect/AIServer/

if [ ! -f "/home/admin/PVDefect/AIServer/ztpanels-haining.tar.gz" ];then
  echo "文件不存在"
  wget ftp://10.123.33.2/PVDefect/AIServer/ztpanels-haining.tar.gz --ftp-user=admin --ftp-password=Ztadmin2020
else
  ifconfig
fi

cd /home/admin/PVDefect/AIServer/bak
rm -rf ztpanels-haining/
cd /home/admin/PVDefect/AIServer/
cp -r ztpanels-haining/ bak/
ps fux | grep python | grep manage.py | grep 8001 |grep -v grep |awk '{print $2}'| xargs kill
cd /home/admin/PVDefect/AIServer/
rm -rf ztpanels-haining

tar -zxvf ztpanels-haining.tar.gz
rm -rf ztpanels-haining.tar.gz

cd /home/admin/PVDefect/AIServer/ztpanels-haining/
nohup /home/admin/anaconda3/envs/yolact_py38_yolov5/bin/python manage.py runserver 0.0.0.0:8001 --noreload  --settings=ztpanels.settings.prod &