# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/31/2021 10:56 AM
# @File:获取电脑所有配置信息
import wmi
import os
w = wmi.WMI()
global list
list=[]
def info():
    list.append("电脑信息")
    for BIOSs in w.Win32_ComputerSystem():
        list.append("电脑名称: %s" %BIOSs.Caption)
        list.append("使 用 者: %s" %BIOSs.UserName)
    for address in w.Win32_NetworkAdapterConfiguration(ServiceName = "e1dexpress"):
        list.append("IP地址: %s" % address.IPAddress[0])
        list.append("MAC地址: %s" % address.MACAddress)
    for BIOS in w.Win32_BIOS():
        list.append("使用日期: %s" %BIOS.Description)
        list.append("主板型号: %s" %BIOS.SerialNumber)
    for processor in w.Win32_Processor():
        list.append("CPU型号: %s" % processor.Name.strip())
    for memModule in w.Win32_PhysicalMemory():
        totalMemSize=int(memModule.Capacity)
        list.append("内存厂商: %s" %memModule.Manufacturer)
        list.append("内存型号: %s" %memModule.PartNumber)
        list.append("内存大小: %.2fGB" %(totalMemSize/1024**3))
    for disk in w.Win32_DiskDrive(InterfaceType = "IDE"):
        diskSize=int(disk.size)
        list.append("磁盘名称: %s" %disk.Caption)
        list.append("磁盘大小: %.2fGB" %(diskSize/1024**3))
    for xk in w.Win32_VideoController():
        list.append("显卡名称: %s" %xk.name)

def main():
    global path
    path= "."
    for BIOSs in w.Win32_ComputerSystem():
        UserNames=BIOSs.Caption
    fileName=path+os.path.sep+UserNames+".txt"
    info()

    #判断文件夹（路径）是否存在
    if not os.path.exists(path):
        print("不存在")
        #创建文件夹（文件路径）
        os.makedirs(path)
        #写入文件信息
        with open(fileName,'w+') as f:
            for li in list:
                print(li)
                l=li+"\n"
                f.write(l)
    else:
        print("存在")
        with open(fileName,'w+') as f:
            for li in list:
                print(li)
                l=li+"\n"
                f.write(l)

main()