# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:6/3/2021 2:07 PM
# @File:more_gpu_deploy

# 多GPU同时运行发布
class BigCutModel(object):
    def __init__(self, base_path=None, cfg=None):
        self.models = {}
        if torch.cuda.is_available():
            for m in range(torch.cuda.device_count()):
                model = Model(cfg=model_path)
                model.load_state_dict(torch.load(weight_path, map_location='cpu')['model'], strict=False)
                model = model.to('cuda:' + str(m))
                model.float().fuse().eval()
                model.half()
                self.models['cuda:' + str(m)] = model
        else:
            model = Model(cfg=model_path)
            model.load_state_dict(torch.load(weight_path, map_location='cpu')['model'], strict=False)
            model = model.to('cpu')
            model.float().fuse().eval()
            model.half()
            self.models['cpu'] = model
        self._model = self.get_one_model()

    @property
    def model(self):
        return next(self._model)

    def get_one_model(self):
        while True:
            for cuda_number, model in self.models.items():
                yield cuda_number, model


class BigCutAlg(object):
    def __init__(self, bigcutmodel):
        self.device, self.model = bigcutmodel.model
        preds = self.model(image, augment=False)[0]