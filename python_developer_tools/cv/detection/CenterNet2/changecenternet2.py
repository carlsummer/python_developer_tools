import torch
import numpy as np
import pickle


def centernet2ModelToCustomerModel(num_class=10, model_path='models/CenterNet2_R50_1x.pth', model_save_dir="models"):
    # 将centernet2的模型转换为自己的训练模型
    pretrained_weights = torch.load(model_path)
    pretrained_weights['iteration'] = 0

    pretrained_weights['model']["roi_heads.box_predictor.0.cls_score.weight"].resize_(num_class + 1, 1024)
    pretrained_weights['model']["roi_heads.box_predictor.0.cls_score.bias"].resize_(num_class + 1)
    pretrained_weights['model']["roi_heads.box_predictor.1.cls_score.weight"].resize_(num_class + 1, 1024)
    pretrained_weights['model']["roi_heads.box_predictor.1.cls_score.bias"].resize_(num_class + 1)
    pretrained_weights['model']["roi_heads.box_predictor.2.cls_score.weight"].resize_(num_class + 1, 1024)
    pretrained_weights['model']["roi_heads.box_predictor.2.cls_score.bias"].resize_(num_class + 1)

    torch.save(pretrained_weights, "{}/CenterNet2_{}.pth".format(model_save_dir, num_class))

if __name__ == '__main__':
    centernet2ModelToCustomerModel(model_path="/home/zengxh/workspace/CenterNet2/projects/CenterNet2/models/CenterNet2_R50_1x.pth", model_save_dir="/home/zengxh/workspace/CenterNet2/projects/CenterNet2/models")