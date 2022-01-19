#!/bin/bash
#定义脚本输出日志文件
logfile="/var/log/Clear_mem/mem_auto_$(date -d "today" +"%Y-%m-%d_%H:%M").log"
if [ ! -d /var/log/Clear_mem ];then #判断日志存放目录是否存在
  mkdir -p /var/log/Clear_mem
else
  echo "======当前系统时间:$(date -d "today" +"%Y-%m-%d_%H:%M:%S")"  >>$logfile
  echo "======当前主机IP地址:$(ifconfig | grep inet | grep -v 172.17.0.1 | grep -v 127.0.0.1 |  grep -v inet6 | grep 10 | awk '{print $2}')" >>$logfile
  echo "======当前主机名:$(hostname)" >>$logfile
fi

#系统分配的区总量
echo "=========Current System Memory Usage==========" >>$logfile
mem_total=`free | grep "Mem:" |awk '{print $2}'`        # 系统总内存
mem_used=`free | grep "Mem:" |awk '{print $3}'`
mem_shared=`free | grep "Mem:" |awk '{print $5}'`
mem_buff=`free | grep "Mem:" |awk '{print $6}'`
mem_usage=`echo "$mem_used+$mem_shared+$mem_buff" | bc`
mem_free=`free | grep "Mem:" | awk '{print $4}'`        # 系统空闲内存
echo "当前系统总内存:$mem_total Kb" >>$logfile
echo "当前系统内存使用量:$mem_usage Kb" >>$logfile
echo "当前系统中空闲内存量:$mem_free Kb" >>$logfile

#计算内存使用率，结果保留两位小数
mem_per=0`echo "scale=2;$mem_usage/$mem_total" |bc`

#判断内存使用率是否超过85%，若超过85%，则执行释放内存动作
mem_waring=0.85  #设置内存使用上限，超过执行释放内存动作
result=`echo "$mem_per > $mem_waring" | bc`
#echo $result
if [ $result = 1 ] ; then
  echo "当前系统内存使用率大于85.00%,内存使用率为:`echo "$mem_per*100"|bc`%,开始执行自动释放内存操作。" >>$logfile
  sync
  echo 3 > /proc/sys/vm/drop_caches
  echo "自动释放内存操作于 $(date -d "today" +"%Y-%m-%d %H:%M:%S") 执行成功。" >>$logfile
else
  echo "当前系统内存使用率低于85%,内存使用率为:`echo "$mem_per*100"|bc`%,无需进行自动释放内存操作。" >>$logfile
fi

#邮件通知功能
domain=@astronergy.com
for name in $(cat Contact_list | grep -v "#")
do
mail -s '内存自动清理邮件' $name$domain < $logfile
done
