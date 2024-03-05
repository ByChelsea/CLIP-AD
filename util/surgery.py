import hashlib
import os
import urllib
import warnings
from typing import Union, List
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


def get_similarity_map(sm, shape, norm=True):
    # min-max norm
    if norm:
        sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])

    # reshape
    side = int(sm.shape[1] ** 0.5)  # square output
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)

    # interpolate
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)

    return sm


def clip_feature_surgery(image_features, text_features, redundant_feats=None, t=2):
    if redundant_feats != None:
        similarity = image_features @ (text_features - redundant_feats).t()

    else:
        # # weights to restrain influence of obvious classes on others
        # prob = image_features[:, :1, :] @ text_features.t()
        # prob = (prob * 2).softmax(-1)
        # w = prob / prob.mean(-1, keepdim=True)
        #
        # # element-wise multiplied features
        # b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
        # feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
        # feats *= w.reshape(1, 1, n_t, 1)
        # redundant_feats = feats.mean(2, keepdim=True) # along cls dim
        # feats = feats - redundant_feats
        #
        # # sum the element-wise multiplied features as cosine similarity
        # similarity = feats.sum(-1)
        similarity = []
        #################### multi-layers fusion ####################
        for idx in range(len(image_features)):
            image_feature = image_features[idx]
            text_feature = text_features
            prob = image_feature[:, :1, :] @ text_feature  # 8, 1, 2
            prob = prob.softmax(-1)  # 8, 1, 2
            # prob = (prob * 2).softmax(-1)
            w = prob / prob.mean(-1, keepdim=True)

            b, n_t, n_i, c = image_feature.shape[0], text_feature.shape[2], image_feature.shape[1], image_feature.shape[
                2]
            feats = image_feature.reshape(b, n_i, 1, c) * text_feature.permute(0, 2, 1).contiguous().reshape(b, 1, n_t,
                                                                                                             c)
            feats *= w.reshape(b, 1, n_t, 1)
            redundant_feats = feats.mean(2, keepdim=True)  # along cls dim
            feats = feats - redundant_feats

            similarity_tmp = feats.sum(-1)
            similarity.append(similarity_tmp)

        similarity = sum(similarity)

    return similarity


# sm shape N_t
def similarity_map_to_points(sm, shape, t=0.8, down_sample=2):
    side = int(sm.shape[0] ** 0.5)
    sm = sm.reshape(1, 1, side, side)

    # down sample to smooth results
    down_side = side // down_sample
    sm = torch.nn.functional.interpolate(sm, (down_side, down_side), mode='bilinear')[0, 0, :, :]
    h, w = sm.shape
    sm = sm.reshape(-1)

    sm = (sm - sm.min()) / (sm.max() - sm.min())
    rank = sm.sort(0)[1]
    scale_h = float(shape[0]) / h
    scale_w = float(shape[1]) / w

    num = min((sm >= t).sum(), sm.shape[0] // 2)
    labels = np.ones(num * 2).astype('uint8')
    labels[num:] = 0
    points = []

    # positives
    for idx in rank[-num:]:
        x = min((idx % w + 0.5) * scale_w, shape[1] - 1)  # +0.5 to center
        # y = min((idx // w + 0.5) * scale_h, shape[0] - 1)
        y = min((torch.div(idx, w, rounding_mode='floor') + 0.5) * scale_h, shape[0] - 1)
        points.append([int(x.item()), int(y.item())])

    # negatives
    for idx in rank[:num]:
        x = min((idx % w + 0.5) * scale_w, shape[1] - 1)
        # y = min((idx // w + 0.5) * scale_h, shape[0] - 1)
        y = min((torch.div(idx, w, rounding_mode='floor') + 0.5) * scale_h, shape[0] - 1)
        points.append([int(x.item()), int(y.item())])

    return points, labels
