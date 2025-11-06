from ultralytics import YOLO
from attention_module import add_attention_to_model

def train_attention_model(
    base_model_path: str = "yolov8m.pt",
    attention_indices: list[int] = None,
    data_path: str = "/content/VOC.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    name: str = "attention_model",
    patience: int = 30,
    cos_lr: bool = True,
    warmup_epochs: float = 2.0,
    lr0: float = 0.01,
    lrf: float = 0.001,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    box: float = 7.5,
    cls: float = 0.5,
    dfl: float = 1.5,
    close_mosaic: int = 10,
    augment: bool = True,
    mixup: float = 0.1,
    copy_paste: float = 0.1,
):
    """
    Train a YOLO model with optional attention modules at specified indices.
    """
    # اضافه کردن attention به مدل
    model = add_attention_to_model(YOLO(base_model_path), attention_indices=attention_indices)
    
    # تنظیمات train
    train_args = {
        "data": data_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "name": name,
        "patience": patience,
        "cos_lr": cos_lr,
        "warmup_epochs": warmup_epochs,
        "lr0": lr0,
        "lrf": lrf,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "box": box,
        "cls": cls,
        "dfl": dfl,
        "close_mosaic": close_mosaic,
        "augment": augment,
        "mixup": mixup,
        "copy_paste": copy_paste
    }

    # آموزش مدل
    results = model.train(**train_args)

    # ارزیابی مدل روی dataset validation/test
    predictions = model.val(data=data_path)
    
    return results, predictions
