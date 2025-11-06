import numpy as np
from skimage.measure import label, regionprops, find_contours

def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))
    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x, y = int(c[0]), int(c[1])
            border[x][y] = 255
    return border


def mask_to_bbox(mask):
    bboxes = []
    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1, y1, x2, y2 = prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]
        bboxes.append([x1, y1, x2, y2])
    return bboxes
