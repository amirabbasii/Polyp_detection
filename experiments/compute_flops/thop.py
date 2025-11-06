
from thop import profile
import torch
from ultralytics import YOLO

# بارگذاری مدل
model_yolo = YOLO("yolov8m.pt")          # مدل YOLOv8-m
model_attn = YOLO("/content/drive/MyDrive/yolo poly8p/yolo_attention_m/weights/best.pt")     # مدل YOLOAttn-V8 (فرضی)

# نمونه ورودی
input_tensor = torch.randn(1, 3, 640, 640)

# تعداد پارامترها و FLOPs برای YOLOv8-m
flops, params = profile(model_yolo.model, inputs=(input_tensor,))
print(f"YOLOv8-m: Params: {params/1e6:.2f}M, FLOPs: {flops/1e9:.2f} GFLOPs")

print("\n\n")
# تعداد پارامترها و FLOPs برای YOLOAttn-V8
flops_attn, params_attn = profile(model_attn.model, inputs=(input_tensor,))
print(f"YOLOAttn-V8: Params: {params_attn/1e6:.2f}M, FLOPs: {flops_attn/1e9:.2f} GFLOPs")