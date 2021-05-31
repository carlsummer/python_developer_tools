import torch

def swinTransformerObjectDetectionToCustomerModel(num_class=2,model_path='mask_rcnn_swin_tiny_patch4_window7.pth',model_save_dir=""):
    pretrained_weights = torch.load(model_path)

    pretrained_weights['state_dict']['roi_head.bbox_head.fc_cls.weight'].resize_(num_class + 1, 1024)
    pretrained_weights['state_dict']['roi_head.bbox_head.fc_cls.bias'].resize_(num_class + 1)
    pretrained_weights['state_dict']['roi_head.bbox_head.fc_reg.weight'].resize_(num_class * 4, 1024)
    pretrained_weights['state_dict']['roi_head.bbox_head.fc_reg.bias'].resize_(num_class * 4)
    pretrained_weights['state_dict']['roi_head.mask_head.conv_logits.weight'].resize_(num_class, 256, 1, 1)
    pretrained_weights['state_dict']['roi_head.mask_head.conv_logits.bias'].resize_(num_class)


    torch.save(pretrained_weights, "{}/mask_rcnn_swin_{}.pth".format(model_save_dir, num_class))

if __name__ == '__main__':
    swinTransformerObjectDetectionToCustomerModel(num_class=10,
                                                  model_path="/home/zengxh/workspace/Swin-Transformer-Object-Detection/mask_rcnn_swin_tiny_patch4_window7.pth",
                                                  model_save_dir="/home/zengxh/workspace/Swin-Transformer-Object-Detection")