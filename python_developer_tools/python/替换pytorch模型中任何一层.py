def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def convert_relu_to_BlurPool(model):
    dyReluchannels = []
    for i, (m, name) in enumerate(zip(model.modules(), model.named_modules())):
        if type(m) is nn.Conv2d:
            dyReluchannels.append({"name": name, "dyrelu":
                DCN(m.in_channels, m.out_channels, kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                    deformable_groups=2).cuda()
                                   })
    for dictsss in dyReluchannels:
        _set_module(model, dictsss["name"][0], dictsss["dyrelu"])
    return model