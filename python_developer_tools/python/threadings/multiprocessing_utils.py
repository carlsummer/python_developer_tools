# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/12/2021 2:45 PM
# @File:multiprocessing_utils
import multiprocessing
from tqdm import tqdm
"""
多线程同时处理批量数据的方法
https://github.com/zhou13/lcnn/blob/88f281ab5421d51a62f1f84f97fea05afbf0c8d8/lcnn/utils.py#L78
例子：
for (key, data_dict_tmp) in data_dict["datasets"].items():
    nameImgs = list(paths.list_images(os.path.join(data_dict_tmp["allDatas"])))
    X_train, X_test_val, _, _ = train_test_split(nameImgs, nameImgs, test_size=0.2, random_state=RANDOM_SEED)

    dict_jsons = parmap(get_dict_json, X_train, 16)
    train_json.extend(dict_jsons)
    dict_jsons = parmap(get_dict_json, X_test_val, 16)
    valid_json.extend(dict_jsons)
"""

def __parallel_handle(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=multiprocessing.cpu_count(), progress_bar=lambda x: x):
    if nprocs == 0:
        nprocs = multiprocessing.cpu_count()
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [
        multiprocessing.Process(target=__parallel_handle, args=(f, q_in, q_out))
        for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()

    try:
        sent = [q_in.put((i, x)) for i, x in enumerate(tqdm(X))]
        [q_in.put((None, None)) for _ in range(nprocs)]
        res = [q_out.get() for _ in progress_bar(range(len(sent)))]
        [p.join() for p in proc]
    except KeyboardInterrupt:
        q_in.close()
        q_out.close()
        raise
    return [x for i, x in sorted(res)]