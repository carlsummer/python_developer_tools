#########################################################################
# Name: InfoView4CentOS
# Version: 1.0
# Author: callmepeanut
# mail: callmepeanut21@gmail.com
# Created Time: Fri 04 Oct 2013 10:04:22 AM CST
#########################################################################
#!/bin/bash

PS3='Please input your choice:'
MENU="HostIp HardwareInfo SystemVersion OpenPorts NonDefault-LoginUsers Exit"

getip()
{
	echo "===============NetworkInfo==============="
	echo "Host ip is: `ifconfig | grep 'inet addr:' | grep Bcast | awk '{print $2}' | awk -F: '{print $2}'`"
	echo ""
}

getVersion()
{
	echo "===============SystemInfo==============="
	echo "Release:         `lsb_release -a | grep Description | cut -d " " -f2-`"
	echo "Kernal Version:  `cat /proc/version | awk '{print $3}'`"
	echo "System Bits:     `getconf LONG_BIT` Bit"
	echo ""
}

getUsers()
{
	echo "===============UsersInfo==============="
	cat /etc/passwd | sed '/nologin/d;/sync/d;/halt/d;/shutdown/d;/root/d'
	echo ""
}

getHardwareInfo()
{
	echo "===============HardwareInfo==============="
	echo "ProductName: `dmidecode | sed -n '24,29p'| grep "Product" | cut -d " " -f3`"
	echo "CPU:         `cat /proc/cpuinfo | grep name | cut -d " " -f3-`"
	echo "PhysicMem:   `cat /proc/meminfo | grep MemTotal | awk '{print $2,$3}'`"
	echo "DiskSpace:   `df -lh | awk '/^\/dev/{ print $2, $5}'`used"
	echo "NetWorkCard: `dmesg | sed -n '440,460p' | awk '/eth0/' | cut -d " " -f4-`"
	echo ""
}

getOpenPorts()
{
	echo "===============OpenPortsInfo==============="
	echo "OpenPorts:"
	echo "`netstat -ntlp | awk '/LISTEN/{printf( "%16s% 16s\n",$4,$7)}'`"
	echo ""
}

select c in $MENU;
do
	case $c in
		HostIp)
			getip;;
		HardwareInfo)
			getHardwareInfo;;
		SystemVersion)
			getVersion;;
		OpenPorts)
			getOpenPorts;;
		NonDefault-LoginUsers)
			getUsers;;
		Exit)
			echo "Bye"
			exit;;
		*)
			echo "Bad Choice"
	esac
done