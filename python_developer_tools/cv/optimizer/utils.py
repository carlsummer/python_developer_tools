def get_current_lr(optimizer):
    return min(g["lr"] for g in optimizer.param_groups)


