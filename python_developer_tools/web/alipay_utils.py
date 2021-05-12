# -*- coding: utf-8 -*-

# pip install python-alipay-sdk==3.0.1
from alipay import AliPay

if __name__ == "__main__":
    alipay = AliPay(
        appid="2021000117656061",  # 应用id
        app_notify_url=None,  # 默认回调url
        app_private_key_string=open("../docs/alipay/keys/private_2048.txt").read(),
        alipay_public_key_string=open("../docs/alipay/keys/alipay_key_2048.txt").read(),
        # 支付宝的公钥，验证支付宝回传消息使用，不是你自己的公钥,
        sign_type="RSA2",  # RSA 或者 RSA2
        debug=True  # 默认False
    )

    order_string = alipay.api_alipay_trade_page_pay(
        subject="测试订单2",
        out_trade_no="3order20210511000014",
        total_amount=0.1,
        return_url="http://127.0.0.1:8000/alipay/return/",
        notify_url="http://127.0.0.1:8000/alipay/return/"
    )

    pay_url = "{}?{data}".format(alipay._gateway,data=order_string)
    print(pay_url)