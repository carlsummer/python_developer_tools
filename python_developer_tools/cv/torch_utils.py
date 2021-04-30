def cuda2cpu(pred):
    # 将cuda的torch变量转为cpu
    if pred.is_cuda:
        pred_cpu = pred.cpu().numpy()
    else:
        pred_cpu = pred.numpy()
    return pred_cpu
