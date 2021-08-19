#### 项目环境搭建
```shell script
# 将six.py copy到C:\Users\zengxh\Documents\software-install\env\itbag\Lib\site-packages\django\utils
# 将boundfield.py copy到 C:\Users\zengxh\Documents\software-install\env\itbag\Lib\site-packages\django\forms\boundfield.py
# Create and run the migrations
python manage.py makemigrations 
# 强制生成makemigration文件
python manage.py makemigrations --empty appname 
python manage.py migrate 
# 同步数据库
python manage.py migrate --run-syncdb
# 设置超级用户
python manage.py createsuperuser
# 启动项目
python manage.py runserver 8000
# 创建新前端对接项目
python manage.py startapp verifycode
# 指定环境变量运行
PYTHONUNBUFFERED=1;DJANGO_SETTINGS_MODULE=ztpanels.settings.dev /home/deploy/anaconda3/envs/yolov5_py38_cu102_conda/bin/python3.8 manage.py migrate
```


#删除关联表数据的时候与之关联也会删除
on_delete = models.CASCADE
#删除关联数据的时候，什么操作也不做
on_delete = models.DO_NOTHIDNG
# 删除关联数据的时候，引发报错
on_delete = models.PROTECT
# 删除关联数据的时候，设置为空
on_delete = models.SET_NULL
#category = models.ForeignKey(ItemsCategory, on_delete=models.SET_NULL, null=True, blank=True,verbose_name="物品类目",help_text="物品类目")

# 删除关联数据的时候，设置为默认值
on_delete = models.SET_DEFAULT
# 删除关联数据
on_delete = models.SET