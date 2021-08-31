#!/bin/bash
#
# Description:
#  Used to get the hardware config information.
#
# History:
#  tcpipstack, 2012/09/04, created.
#

#------------------------------------------------------------------------------------
fn_get_cpu_info()
{
echo -e "\n CPU Num is: "`grep -c 'model name' /proc/cpuinfo`
echo "`cat /proc/cpuinfo | grep processor | wc -l`"
#echo `cat /proc/cpuinfo | grep 'model name' | sed 's'`
}

fn_get_disk_info()
{
echo -e "\n Disk Information: "
for x in `df -h | grep /dev | awk '{print $5 "-" $6 "-" $2 "-" $4}' | sed 's/%//g'`
do
disk_status=(${x//"-"/" "})
echo "Disk Directory ${disk_status[1]} DiskTotal=${disk_status[2]} DiskUsed=${disk_status[3]}"
done
}

fn_get_mem_info()
{
 MemTotal=`free -m | grep Mem | awk '{print  $2}'`
 echo -e "\n Memory is: ${MemTotal} MB "
}

#------------------------------------------------------------------------------------
echo -e "\n -----------This Computer's Hardware Config Information is: -----------\n"
fn_get_disk_info
fn_get_cpu_info
fn_get_mem_info
echo -e "\n -----------End -----------\n"