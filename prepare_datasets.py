from mask_conversion import mask_to_bbox, mask_to_border
import os
import cv2
import json
import shutil
import numpy as np
import random
from skimage import io
from ultralytics import YOLO



def create_data_directories():
    if not os.path.exists('data'):
        for folder in ['images', 'labels']:
            for split in ['train', 'val', 'test']:
                os.makedirs(f'data/{folder}/{split}', exist_ok=True)


# ==========================================================
# Dataset preparation
# ==========================================================

def prepare_kvasir_seg():
    # ✅ Reproducible shuffle
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    images = os.listdir("/content/Kvasir-SEG/images")
    random.shuffle(images)
    print(f"Kvasir shuffled: {len(images)} images")

    f = open('/content/Kvasir-SEG/kavsir_bboxes.json')
    json_file = json.load(f)
    create_data_directories()

    for i, name in enumerate(json_file.keys()):
        img_width = json_file[name]['width']
        img_height = json_file[name]['height']

        ans = ""
        for bbox in json_file[name]['bbox']:
            w = bbox['xmax'] - bbox['xmin']
            h = bbox['ymax'] - bbox['ymin']
            x_center = bbox['xmin'] + w / 2
            y_center = bbox['ymin'] + h / 2

            x_center /= img_width
            y_center /= img_height
            w /= img_width
            h /= img_height

            ans += f'0 {x_center} {y_center} {w} {h}\n'

        # ✅ Keep original index-based split
        if i < 900:
            section = "train"
        elif i < 900 + 90:
            section = "val"
        else:
            section = "test"

        src = f'/content/Kvasir-SEG/images/{name}.jpg'
        dst = f'data/images/{section}/Kavir_{name}.jpg'
        shutil.copy(src, dst)

        with open(f'data/labels/{section}/Kavir_{name}.txt', 'w') as f:
            f.write(ans)


def prepare_cvc_clinicdb():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    images = os.listdir("/content/CVC-ClinicDB/Original")
    random.shuffle(images)
    print(f"CVC-ClinicDB shuffled: {len(images)} images")

    create_data_directories()

    for i, name in enumerate(images):
        image = io.imread(f'/content/CVC-ClinicDB/Original/{name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(f'/content/CVC-ClinicDB/Ground Truth/{name}')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        bboxes = mask_to_bbox(mask)
        ans = ""
        for x, y, x2, y2 in bboxes:
            w, h = x2 - x, y2 - y
            x, y = x + w/2, y + h/2
            x /= image.shape[1]
            y /= image.shape[0]
            w /= image.shape[1]
            h /= image.shape[0]
            ans += f'0 {x} {y} {w} {h}\n'

        if i < 550:
            section = "train"
        elif i < 550 + 55:
            section = "val"
        else:
            section = "test"

        out_img = f'data/images/{section}/CVC-ClinicDB_{name.replace("tif", "jpg")}'
        out_lbl = f'data/labels/{section}/CVC-ClinicDB_{name.replace(".tif", ".txt")}'
        cv2.imwrite(out_img, image)
        with open(out_lbl, 'w') as f:
            f.write(ans)


def prepare_etis():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    images = os.listdir("/content/ETIS-LaribPolypDB/ETIS-LaribPolypDB")
    random.shuffle(images)
    print(f"ETIS shuffled: {len(images)} images")

    create_data_directories()

    for i, name in enumerate(images):
        image = io.imread(f'/content/ETIS-LaribPolypDB/ETIS-LaribPolypDB/{name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(f'/content/ETIS-LaribPolypDB/Ground Truth/p{name}')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        bboxes = mask_to_bbox(mask)
        ans = ""
        for x, y, x2, y2 in bboxes:
            w, h = x2 - x, y2 - y
            x, y = x + w/2, y + h/2
            x /= image.shape[1]
            y /= image.shape[0]
            w /= image.shape[1]
            h /= image.shape[0]
            ans += f'0 {x} {y} {w} {h}\n'

        if i < 100:
            section = "train"
        elif i < 100 + 10:
            section = "val"
        else:
            section = "test"

        out_img = f'data/images/{section}/ETIS_{name.replace("tif", "jpg")}'
        out_lbl = f'data/labels/{section}/ETIS_{name.replace(".tif", ".txt")}'
        cv2.imwrite(out_img, image)
        with open(out_lbl, 'w') as f:
            f.write(ans)


def prepare_cvc_colondb():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    path = '/content/CVC-ColonDB'
    images = os.listdir(f'{path}/CVC-ColonDB/images')
    random.shuffle(images)
    print(f"CVC-ColonDB shuffled: {len(images)} images")

    create_data_directories()

    for i, name in enumerate(images):
        image = io.imread(f'{path}/CVC-ColonDB/images/{name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(f'{path}/CVC-ColonDB/masks/{name}')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        bboxes = mask_to_bbox(mask)
        ans = ""
        for x, y, x2, y2 in bboxes:
            w, h = x2 - x, y2 - y
            x, y = x + w/2, y + h/2
            x /= image.shape[1]
            y /= image.shape[0]
            w /= image.shape[1]
            h /= image.shape[0]
            ans += f'0 {x} {y} {w} {h}\n'

        if i < 300:
            section = "train"
        elif i < 300 + 30:
            section = "val"
        else:
            section = "test"

        out_img = f'data/images/{section}/CVC-ColonDB_{name.replace("tif", "jpg")}'
        out_lbl = f'data/labels/{section}/CVC-ColonDB_{name.replace(".tif", ".txt")}'
        cv2.imwrite(out_img, image)
        with open(out_lbl, 'w') as f:
            f.write(ans)
def prepare_endoscene():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    path = '/content/EndoScene'
    images = os.listdir(f'{path}/images')
    random.shuffle(images)
    print(f"EndoScene shuffled: {len(images)} images")

    create_data_directories()

    for i, name in enumerate(images):
    
        image = io.imread(f'{path}/images/{name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
        mask = cv2.imread(f'{path}/masks/{name}')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        bboxes = mask_to_bbox(mask)
        ans = ""
        for x, y, x2, y2 in bboxes:
            w, h = x2 - x, y2 - y
            x, y = x + w / 2, y + h / 2
            x /= image.shape[1]
            y /= image.shape[0]
            w /= image.shape[1]
            h /= image.shape[0]
            ans += f'0 {x} {y} {w} {h}\n'

        if i < 40:
            section = "train"
        elif i < 44:
            section = "val"
        else:
            section = "test"


        out_img = f'data/images/{section}/EndoScene_{name.replace("tif", "jpg")}'
        out_lbl = f'data/labels/{section}/EndoScene_{name.replace(".tif", ".txt")}'

        cv2.imwrite(out_img, image)
        with open(out_lbl, 'w') as f:
            f.write(ans)

    print("✅ EndoScene dataset prepared successfully.")


# ==========================================================
# YAML config + main
# ==========================================================
def save_yaml():
    conf = (
        "train: /content/data/images/train\n\n"
        "val: /content/data/images/test\n\n"
        "test: /content/data/images/test\n\n"
        "nc: 1\n\n"
        "names: ['x']"
    )
    with open("VOC.yaml", "w") as f:
        f.write(conf)


def prepare_all_datasets():
    prepare_kvasir_seg()
    prepare_cvc_clinicdb()
    prepare_etis()
    prepare_cvc_colondb()
    prepare_endoscene()
    save_yaml()
