import torch.nn as nn
import segmentation_models_pytorch as smp
import torch
from torch.cuda.amp import autocast

class seg_qyl(nn.Module):
    def __init__(self, model_name, n_class):
        super(seg_qyl,self).__init__()
        self.model = smp.Unet(# UnetPlusPlus
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights='imagenet',     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=n_class,                      # model output channels (number of classes in your dataset)
                decoder_attention_type='se'
            )

        # self.deep_stem_layer = self._make_stem_layer(self.model.encoder.conv1.in_channels, self.model.encoder.conv1.out_channels,
        #                           self.model.encoder._norm_layer).to(
        #         torch.device(self.model.encoder.conv1.weight.device))

    def _make_stem_layer(self, in_channels, stem_channels, norm_layer):
        """Make stem layer for ResNet. self.deep_stem:"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            norm_layer(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                stem_channels // 2,
                stem_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            norm_layer(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                stem_channels // 2,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            norm_layer(stem_channels),
            nn.ReLU(inplace=True))

    @autocast()
    def forward(self, x):
        stages = self.model.encoder.get_stages()
        # stages[1] = self.deep_stem_layer
        features = []
        for i in range(self.model.encoder._depth + 1):
            x = stages[i](x)
            features.append(x)

        # features = self.model.encoder(x)
        decoder_output = self.model.decoder(*features)

        masks = self.model.segmentation_head(decoder_output)

        if self.model.classification_head is not None:
            labels = self.model.classification_head(features[-1])
            return masks, labels

        return masks