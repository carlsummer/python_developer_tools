# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/18/2021 9:05 AM
# @File:pickle_utils
import pickle


def write_pkl(path="test.pkl",data={"test":"test 测试"}):
    # data = {'image': image, 'lines': H["lines"][0], 'scores': H["score"][0], "w0": w0, "h0": h0, "pad": pad,
    #         "imshape": im.shape, "nlines": nlines}
    with open(path,"wb") as f:
        pickle.dump(data, f)

def read_pkl(path="test.pkl"):
    with open(path, 'rb') as f:
        data = pickle.load(f,encoding="utf-8")
    return data

if __name__ == '__main__':
    write_pkl()
    data = read_pkl()
    print(data)