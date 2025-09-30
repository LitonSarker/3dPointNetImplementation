#metrcs_pointcloud_cpu.py
#metrics for point cloud segmentation on CPU
import numpy as np

def confusion_matrix(num_classes, gt, pred):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    mask = (gt >= 0) & (gt < num_classes)
    gt = gt[mask]
    pred = pred[mask]
    for g, p in zip(gt, pred):
        cm[g, p] += 1
    return cm

def scores_from_cm(cm):
    eps = 1e-9
    correct = np.trace(cm)
    total = cm.sum()
    oa = correct / (total + eps)
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    denom = tp + fp + fn + 1e-9
    iou = tp / denom
    miou = float(np.mean(iou))
    acc_per_class = tp / (tp + fn + 1e-9)
    macc = float(np.mean(acc_per_class))
    return {
        'OA': float(oa),
        'mAcc': macc,
        'mIoU': miou,
        'IoU_per_class': [float(x) for x in iou],
        'Acc_per_class': [float(x) for x in acc_per_class],
    }
