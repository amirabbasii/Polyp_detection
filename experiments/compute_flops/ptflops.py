import torch
from ptflops import get_model_complexity_info
from ultralytics import YOLO

base_model = YOLO("yolov8m.pt").model
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(base_model, (3, 640, 640), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
print(f"MACs: {macs}, Params: {params}")




proposed_model = YOLO("yolo_attention_m/weights/best.pt").model
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(proposed_model, (3, 640, 640), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
print(f"MACs: {macs}, Params: {params}")

