# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/29/2021 2:41 PM
# @File:setup
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding="utf-8") as f:
    README = f.read()

setup(
    author='zengxiaohui',  # 可以用来指定该package的作者信息。
    name='python_developer_tools',  # 是该package的名字。该名字可以由字母，数字，_和-组成。并且这个名字不能与其他已经上传至pypi.org的项目相同
    version='0.0.2',  # 是当前package的版本。关于版本的详细信息请参考
    author_email="2362651588@qq.com",  # 可以用来指定该package的作者信息。
    description='python developer tools',
    long_description=README,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Topic :: System :: Networking :: Monitoring",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    url='https://github.com/carlsummer/python_developer_tools',
    packages=find_packages(),  # find_packages()告诉Distutils需要处理那些包（包含__init__.py的文件夹）
    # package_dir告诉Distutils哪些目录下的文件被映射到哪个源码包。一个例子：package_dir = {'': 'lib'}，表示“root package”中的模块都在lib目录中。
    package_dir={"cv": "cv",
                 "files": "files",
                 "python": "python",
                 "web": "web", },
    # package_data={"docs": ["/docs/keys/*.txt"]},
    # data_files=[
    #     ('docs', ['python_developer_tools/docs/keys/*.txt',]),
    # ],
    install_requires=["numpy", "scipy", "matplotlib"],

)
