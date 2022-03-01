import torch

# make training more stable
class WeightEMA:
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.weight_decacy = 0.0004

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param) # 是把ema_param的产生copy 给param

    def step(self):
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:  # model weights only!
                ema_param.mul_(self.alpha)
                ema_param.add_(param * (1 - self.alpha))
                # apply weight
                param.mul_((1 - self.weight_decacy))