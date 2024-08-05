import torch
from torchvision.ops import box_iou


def calculate_metrics(pred_boxes, pred_labels, true_boxes, true_labels, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0

    for true_box, true_label in zip(true_boxes, true_labels):
        matched = False
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            iou = box_iou(pred_box.unsqueeze(0), true_box.unsqueeze(0)).item()
            if iou >= iou_threshold and pred_label == true_label:
                tp += 1
                matched = True
                break
        if not matched:
            fn += 1

    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        matched = False
        for true_box, true_label in zip(true_boxes, true_labels):
            iou = box_iou(pred_box.unsqueeze(0), true_box.unsqueeze(0)).item()
            if iou >= iou_threshold and pred_label == true_label:
                matched = True
                break
        if not matched:
            fp += 1

    return tp, fp, fn

