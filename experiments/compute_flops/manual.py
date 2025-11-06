
import torch.nn as nn
from ultralytics import YOLO

from attention_module import add_attention_to_model

# -------------------------------
# توابع محاسبه FLOPs دقیق
# -------------------------------

def conv2d_flops(layer, input_size):
    batch, Cin, H_in, W_in = input_size
    Cout = layer.out_channels
    KH, KW = layer.kernel_size
    stride_h, stride_w = layer.stride
    H_out = (H_in + 2*layer.padding[0] - KH)//stride_h + 1
    W_out = (W_in + 2*layer.padding[1] - KW)//stride_w + 1
    groups = layer.groups
    flops = 2 * H_out * W_out * Cin * KH * KW * Cout // groups
    return flops, (batch, Cout, H_out, W_out)

def linear_flops(layer, input_size):
    batch, in_features = input_size
    out_features = layer.out_features
    flops = 2 * in_features * out_features
    return flops, (batch, out_features)

def pool_flops(layer, input_size):
    batch, Cin, H_in, W_in = input_size

    # بررسی kernel_size
    if isinstance(layer.kernel_size, int):
        KH = KW = layer.kernel_size
    else:
        KH, KW = layer.kernel_size

    # بررسی stride
    if isinstance(layer.stride, int):
        stride_h = stride_w = layer.stride
    else:
        stride_h, stride_w = layer.stride

    H_out = (H_in - KH)//stride_h + 1
    W_out = (W_in - KW)//stride_w + 1

    flops = KH * KW * Cin * H_out * W_out  # هر پیکسل یک عملیات
    return flops, (batch, Cin, H_out, W_out)

def upsample_flops(layer, input_size):
    batch, Cin, H_in, W_in = input_size
    scale = layer.scale_factor if isinstance(layer.scale_factor, int) else 1
    H_out = int(H_in * scale)
    W_out = int(W_in * scale)
    # ضرب ساده هر پیکسل
    flops = Cin * H_out * W_out
    return flops, (batch, Cin, H_out, W_out)

def activation_flops(layer, input_size):
    batch, Cin, H, W = input_size
    flops = Cin * H * W  # فرض یک عملیات per pixel
    return flops, input_size

# -------------------------------
# محاسبه FLOPs برای کل مدل
# -------------------------------

def compute_flops(model, input_size=(1,3,640,640)):
    flops_total = 0
    size = input_size
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            flops, size = conv2d_flops(layer, size)
            flops_total += flops
        elif isinstance(layer, nn.Linear):
            if len(size) > 2:
                size = (size[0], size[1]*size[2]*size[3])
            flops, size = linear_flops(layer, size)
            flops_total += flops
        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            flops, size = pool_flops(layer, size)
            flops_total += flops
        elif isinstance(layer, nn.Upsample):
            flops, size = upsample_flops(layer, size)
            flops_total += flops
        elif isinstance(layer, (nn.ReLU, nn.SiLU, nn.Sigmoid, nn.Tanh)):
            flops, size = activation_flops(layer, size)
            flops_total += flops
    return flops_total

# -------------------------------
# اجرای محاسبه روی مدل‌ها
# -------------------------------

# YOLOv8-m
model_yolo = YOLO("yolov8m.pt").model
flops_yolo = compute_flops(model_yolo)
print(f"YOLOv8-m FLOPs: {flops_yolo/1e9:.2f} GFLOPs")

# YOLOAttn-V8 (مدل توسعه یافته)
from copy import deepcopy
model_attn = deepcopy(YOLO("yolov8m.pt"))
model_attn = add_attention_to_model(model_attn).model
flops_attn = compute_flops(model_attn)
print(f"YOLOAttn-V8 FLOPs: {flops_attn/1e9:.2f} GFLOPs")

increase = (flops_attn - flops_yolo)/flops_yolo*100
print(f"Computation increase: {increase:.2f}%")
