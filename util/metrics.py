from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from tabulate import tabulate
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc, roc_auc_score, average_precision_score
from skimage import measure
import cv2
import os
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import functional as F

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float64)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def calculate_metrics(results, logger, obj_list):
    table_ls, auroc_sp_ls, f1_sp_ls, ap_sp_ls, auroc_px_ls, f1_px_ls, aupro_ls, ap_px_ls = [], [], [], [], [], [], [], []
    for obj in tqdm(obj_list, desc="Evaluating"):
        table = [obj]
        gt_px, pr_px, gt_sp, pr_sp0, pr_sp1, pr_sp = [], [], [], [], [], []
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                if len(results['gt_px']) != 0:
                    gt_px.append(results['gt_px'][idxes])
                pr_px.append(results['pr_px'][idxes])
                if len(results['gt_sp']) != 0:
                    gt_sp.append(results['gt_sp'][idxes])
                pr_sp0.append(results['pr_sp0'][idxes])
                pr_sp1.append(results['pr_sp1'][idxes])
        if len(gt_px) != 0:
            gt_px = np.array(gt_px)
        if len(gt_sp) != 0:
            gt_sp = np.array(gt_sp)
        pr_px = np.array(pr_px)
        pr_sp0 = normalize(np.array(pr_sp0))
        pr_sp1 = normalize(np.array(pr_sp1))
        pr_sp = pr_sp0 + pr_sp1

        # pixel
        if len(gt_px) != 0:
            auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
            # aupro = cal_pro_score(gt_px, pr_px)
            aupro = 0.0
            ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
            precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
            f1_scores = (2 * precisions * recalls) / (precisions + recalls)
            f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        # sample
        if len(gt_sp) != 0:
            auroc_sp = roc_auc_score(gt_sp, pr_sp)
            ap_sp = average_precision_score(gt_sp, pr_sp)
            precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
            f1_scores = (2 * precisions * recalls) / (precisions + recalls)
            f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])

        # construct a table
        metrics = []
        if len(gt_px) != 0 and len(gt_sp) != 0:
            metrics = [auroc_px, f1_px, ap_px, aupro, auroc_sp, f1_sp, ap_sp]
        elif len(gt_px) == 0:
            metrics = [auroc_sp, f1_sp, ap_sp]
        else:
            metrics = [auroc_px, f1_px, ap_px, aupro]
            
        table.extend([f"{np.round(value * 100, decimals=1):.1f}" for value in metrics])

        table_ls.append(table)
        if len(gt_px) != 0:
            auroc_px_ls.append(auroc_px)
            f1_px_ls.append(f1_px)
            ap_px_ls.append(ap_px)
            aupro_ls.append(aupro)
        if len(gt_sp) != 0:
            auroc_sp_ls.append(auroc_sp)
            f1_sp_ls.append(f1_sp)
            ap_sp_ls.append(ap_sp)

    # logger
    if len(gt_px) != 0 and len(gt_sp) != 0:
        table_ls.append(['mean'] + [f"{np.round(np.mean(lst) * 100, decimals=1):.1f}" for lst in
                                    [auroc_px_ls, f1_px_ls, ap_px_ls, aupro_ls, auroc_sp_ls, f1_sp_ls, ap_sp_ls]])
        results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro', 'auroc_sp',
                                              'f1_sp', 'ap_sp'], tablefmt="pipe")
    elif len(gt_px) == 0:
        table_ls.append(['mean'] + [f"{np.round(np.mean(lst) * 100, decimals=1):.1f}" for lst in
                                    [auroc_sp_ls, f1_sp_ls, ap_sp_ls]])
        results = tabulate(table_ls, headers=['objects', 'auroc_sp', 'f1_sp', 'ap_sp'], tablefmt="pipe")
    else:
        table_ls.append(['mean'] + [f"{np.round(np.mean(lst) * 100, decimals=1):.1f}" for lst in
                                    [auroc_px_ls, f1_px_ls, ap_px_ls, aupro_ls]])
        results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro'], tablefmt="pipe")

    logger.info("\n%s", results)


def visualization(results, obj_list, img_size, save_path):
    for obj in obj_list:
        paths = []
        pr_px = []
        cls_names = []
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                paths.append(results['img_path'][idxes])
                pr_px.append(results['pr_px'][idxes])
                cls_names.append(results['cls_names'][idxes])
        pr_px = normalize(np.array(pr_px))
        # draw
        for idx in range(len(paths)):
            path = paths[idx]
            cls = path.split('/')[-2]
            filename = path.split('/')[-1]
            vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
            mask = normalize(pr_px[idx])
            vis = apply_ad_scoremap(vis, mask) 
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
            save_vis = os.path.join(save_path, 'imgs', cls_names[idx], cls)
            os.makedirs(save_vis, exist_ok=True)
            cv2.imwrite(os.path.join(save_vis, filename), vis)
