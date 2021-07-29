# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/23/2021 9:17 AM
# @File:read_yaml
import pprint

from python_developer_tools.python.yamls.box import Box
import yaml

# 第一种读取方法
# C is a dict storing all the configuration
C = Box()
# shortcut for C.model
M = Box()
config_file = "wireframe.yaml"
C.update(C.from_yaml(filename=config_file))
M.update(C.model)
pprint.pprint(C, indent=4)  # pprint打印的比print打印的结构更加完整


# 第二种读取方法
def parseYamlCfg(cfg_path):
    return yaml.load(open(cfg_path, 'r', encoding='utf-8').read(), Loader=yaml.FullLoader)


cfg = parseYamlCfg(config_file)
pprint.pprint(C, indent=4)

# 将yaml进行保存
# C.to_yaml(osp.join(outdir, "config.yaml"))