# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/22/2021 9:44 PM
# @File:df
from qcloudsms_py import SmsSingleSender
from qcloudsms_py.httpclient import HTTPError
import random
import ssl

from python_developer_tools.web.services_utils import make_code

ssl._create_default_https_context = ssl._create_unverified_context

appid = '1400525419'  # 准备工作中的SDK AppID，类型：int
appkey = '2c894050f230f6df92abac6da8048e57'  # 准备工作中的App Key，类型：str
sign = 'CollectInto'  # 准备工作中的应用签名，类型：str

def send_msg(phone_num):
    ssender = SmsSingleSender(appid, appkey)
    try:
        # parms参数类型为list
        rzb = ssender.send_with_param(86, phone_num, 968642, [make_code()],
                                      sign=sign, extend='', ext='')
        print(rzb)
    except HTTPError as http:
        print("HTTPError", http)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    phone_num = '18013634236'
    send_msg(phone_num)