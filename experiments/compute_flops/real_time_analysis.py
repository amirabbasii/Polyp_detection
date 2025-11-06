import time

import torch
from ultralytics import YOLO

from attention_module import add_attention_to_model
from experiments.compute_flops.manual import compute_flops


input_tensor = torch.randn(1, 3, 640, 640)
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
# مدل روی GPU
device = "cuda"
model_yolo.model.to(device)
input_tensor = input_tensor.to(device)

# warm-up
for _ in range(10):
    _ = model_yolo.model(input_tensor)

# اندازه‌گیری زمان
start = time.time()
for _ in range(100):
    _ = model_yolo.model(input_tensor)
end = time.time()

avg_time = (end - start) / 100  # زمان میانگین هر تصویر
fps = 1 / avg_time

print(f"YOLOv8-m: {avg_time*1000:.2f} ms/image, {fps:.2f} FPS")



model_attn.model.to(device)
input_tensor = input_tensor.to(device)

# warm-up
for _ in range(10):
    _ = model_attn.model(input_tensor)

# اندازه‌گیری زمان
start = time.time()
for _ in range(100):
    _ = model_attn.model(input_tensor)
end = time.time()

avg_time = (end - start) / 100  # زمان میانگین هر تصویر
fps = 1 / avg_time

print(f"YOLOv8-m: {avg_time*1000:.2f} ms/image, {fps:.2f} FPS")



