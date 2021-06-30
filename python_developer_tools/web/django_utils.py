# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:6/28/2021 9:07 AM
# @File:django_utils

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