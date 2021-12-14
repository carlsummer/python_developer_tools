# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# added by Anaconda3 installer
export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME=enp6s0
export NCCL_IB_DISABLE=1
export PATH="/home/zengxh/anaconda3/bin:$PATH"


