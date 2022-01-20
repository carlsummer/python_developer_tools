import torchvision
import torch

# 初始化模型
model = torchvision.models.shufflenet_v2_x0_5(pretrained=True).cuda()
# 获取图片
input2 = torch.randn(144, 3, 960, 960).cuda()


# 下面的代码会自己释放显存
with torch.no_grad():
    aa= torch.cuda.memory_cached()
    import gc
    out = model(input2)
    # 然后释放，下面这段代码不放开也行
    # input2 = input2.cpu()
    # del input2
    # del model

    # 这里虽然将上面的显存释放了，但是我们通过Nvidia-smi命令看到显存依然在占用
    torch.cuda.empty_cache()
    # 只有执行完上面这句，显存才会在Nvidia-smi中释放

    gc.collect()
    bb=torch.cuda.memory_cached()


# 下面的代码不会自己释放显存
out = model(input2)
