# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/30/2021 8:51 AM
# @File:threading_utils
import threading


class ProcessThread1(threading.Thread):
    def __init__(self, idx, worker_num, images, out_img, out_h, out_w, device, input_w=320, input_h=640):
        threading.Thread.__init__(self)
        self.images = images
        self.out_img = out_img
        self.out_h = out_h
        self.out_w = out_w
        self.input_w = input_w
        self.input_h = input_h
        self.idx = idx
        self.worker_num = worker_num
        self.device = device
        # self.pixels = self.input_h * self.input_w * 3

    def preprocess_image(self, raw_bgr_image, num):
        image = cv2.resize(raw_bgr_image, (self.input_w, self.input_h))
        image = image.astype(np.float32)
        image /= 255.0
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        self.out_img[self.idx + num] = torch.from_numpy(image.copy()).to(self.device)
        # self.out_img.append(torch.from_numpy(image.copy()).to(self.device))
        return image

    def run(self):
        # print(self.idx, ' start ')
        for i in range(self.worker_num):
            self.preprocess_image(self.images[self.idx + i], i)
            self.out_h[self.idx + i] = self.images[self.idx + i].shape[0]
            self.out_w[self.idx + i] = self.images[self.idx + i].shape[1]
        # print(self.idx, ' end ')

if __name__ == '__main__':
    batch_size = 144
    thread_num = num_workers
    batch_thread = batch_size // thread_num
    batch_origin_h = torch.zeros(batch_size).to(device)
    batch_origin_w = torch.zeros(batch_size).to(device)
    threads = []
    inputs = [0 for _ in range(batch_size)]
    for i in range(thread_num):
        thread = ProcessThread1(i * batch_thread, batch_thread, imgs,
                               inputs, batch_origin_h, batch_origin_w, device,
                               input_w=imgsize[1], input_h=imgsize[0])
        thread.start()
        threads.append(thread)
    for thr in threads:
        thr.join()
    # thr.join完就已经把值返回到了inputs了
    inputs = torch.cat(inputs)