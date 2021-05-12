# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/12/2021 10:27 AM
# @File:requests_utils

def get_scheme_host(HttpRequest):
    # 返回http://127.0.0.1:8000
    return '{scheme}://{host}'.format(
        scheme=HttpRequest.scheme,
        host=HttpRequest._get_raw_host(),
    )
