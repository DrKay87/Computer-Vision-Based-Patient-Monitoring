import cv2
import numpy as np
import random

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def center_distance(b1, b2):
    return np.linalg.norm(bbox_center(b1) - bbox_center(b2))

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def center_distance(b1, b2):
    return np.linalg.norm(bbox_center(b1) - bbox_center(b2))

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter)

def compute_union_box(boxes):

    valid_boxes = [box for box in boxes if box is not None]

    x1 = min(b[0] for b in valid_boxes)
    y1 = min(b[1] for b in valid_boxes)
    x2 = max(b[2] for b in valid_boxes)
    y2 = max(b[3] for b in valid_boxes)

    return [x1, y1, x2, y2]

def normalize_bbox(boxes, ref_box, area_thresh=0.6):
    normalized = []

    rx1, ry1, rx2, ry2 = ref_box
    ref_area = (rx2 - rx1) * (ry2 - ry1)

    for i, box in enumerate(boxes):
        
        if box is None:
            normalized.append(None)
            continue

        x1, y1, x2, y2 = box
        curr_area = (x2 - x1) * (y2 - y1)

        if curr_area < area_thresh * ref_area:
            normalized.append(ref_box)
            print(f'box: {i}, box area: {curr_area}, ref bx area: {ref_area}')
        else:
            normalized.append(box)

    return normalized


def smooth_boxes(boxes, ref_box, window=5):
    
    boxes = normalize_bbox(boxes, ref_box, area_thresh=0.6)

    smoothed = []
    for i in range(len(boxes)):
          
        win = [
            boxes[j] for j in range(max(0, i-window), min(len(boxes), i+window))
            if boxes[j] is not None
        ]
        if not win:
            smoothed.append(None)
        else:
            smoothed.append(np.mean(win, axis=0).tolist())

        
    return smoothed

def expand_box(box, img_w, img_h, scale_x=0.25, scale_y=0.25):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_w = w * (1 + scale_x)
    new_h = h * (1 + scale_y)

    nx1 = int(max(0, cx - new_w / 2))
    ny1 = int(max(0, cy - new_h / 2))
    nx2 = int(min(img_w, cx + new_w / 2))
    ny2 = int(min(img_h, cy + new_h / 2))

    return [nx1, ny1, nx2, ny2]