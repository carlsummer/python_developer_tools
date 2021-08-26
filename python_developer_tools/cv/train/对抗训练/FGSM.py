# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/26/2021 8:41 AM
# @File:FGSM
import torch
# FGSM attack code
# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/fba7866856a418520404ba3a11142335/fgsm_tutorial.ipynb#scrollTo=JxUKkTqn5_9O
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image