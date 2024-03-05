import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import KernelPCA


def pca(X, k):
  n_samples, n_features = X.shape
  mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
  norm_X = X - mean
  scatter_matrix = np.dot(np.transpose(norm_X),norm_X)
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
  eig_pairs.sort(reverse=True)
  feature = np.array([ele[1] for ele in eig_pairs[:k]])
  data = np.dot(norm_X,np.transpose(feature))
  return data


def encode_text_with_prompt_ensemble(model, objs, tokenizer, device, dataset, rep_vec='mean'):
    prompt_normal = ['flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect',
                     '{} without damage', '{} without scratch', '{} without crack', '{} without contamination']
    prompt_abnormal = ['damaged {}', 'imperfect {}', 'blemished {}', 'broken {}', '{} with flaw', '{} with defect',
                       '{} with damage', '{} with scratch', '{} with crack', '{} with contamination']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a cropped photo of the {}.', 'a cropped photo of a {}.',
                        'a close-up photo of a {}.', 'a close-up photo of the {}.',
                        'a bright photo of a {}.', 'a bright photo of the {}.',
                        'a dark photo of the {}.', 'a dark photo of a {}.',
                        'a jpeg corrupted photo of a {}.', 'a jpeg corrupted photo of the {}.',
                        'a blurry photo of the {}.', 'a blurry photo of a {}.',
                        'a photo of the {}.', 'a photo of a {}.',
                        'a photo of a large {}.', 'a photo of the small {}.',
                        'a photo of the large {}.', 'a photo of a small {}.',
                        'a photo of the {} for visual inspection.', 'a photo of a {} for visual inspection.',
                        'a photo of the {} for anomaly detection.', 'a photo of a {} for anomaly detection.']

    text_prompts = {}
    for obj in objs:
        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = tokenizer(prompted_sentence).to(device)
            class_embeddings = model.encode_text(prompted_sentence)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if rep_vec == 'mean':
                class_embedding = class_embeddings.mean(dim=0)
            elif rep_vec == 'pca':
                mean = class_embeddings.mean(dim=0)
                class_embedding = pca(class_embeddings.permute(1, 0).cpu().numpy().astype(np.float32), 1)
                class_embedding = torch.tensor(class_embedding, dtype=torch.float16).squeeze().to(device)
                product = torch.dot(mean, class_embedding)
                if product < 0:
                    class_embedding = -class_embedding
            elif rep_vec == 'kde':
                class_embeddings = class_embeddings.cpu().numpy()
                kde = KernelDensity(kernel='gaussian', bandwidth=0.3)
                kde.fit(class_embeddings)
                log_densities = kde.score_samples(class_embeddings)[:, None]
                class_embedding = np.sum(np.exp(log_densities) * class_embeddings, axis=0) / np.sum(np.exp(log_densities))
                class_embedding = torch.tensor(class_embedding, dtype=torch.float16).squeeze()
            elif rep_vec == 'dbscan':
                data = class_embeddings.cpu().numpy()
                if dataset == 'visa':
                    dbscan = DBSCAN(eps=1.5, min_samples=25)
                else:
                    dbscan = DBSCAN(eps=0.5, min_samples=15)
                dbscan.fit(data)
                labels = dbscan.labels_
                label_counts = np.bincount(labels[labels >= 0])
                label = np.argmax(label_counts)
                class_embedding = np.mean(data[labels == label], axis=0)
                class_embedding = torch.tensor(class_embedding, dtype=torch.float16).squeeze()
            elif rep_vec == 'mean_shift':
                data = class_embeddings.cpu().numpy()
                mean_shift = MeanShift(bandwidth=2)
                mean_shift.fit(data)
                labels = mean_shift.labels_
                label_counts = np.bincount(labels[labels >= 0])
                label = np.argmax(label_counts)
                class_embedding = np.mean(data[labels == label], axis=0)
                class_embedding = torch.tensor(class_embedding, dtype=torch.float16).squeeze()
            else:
                raise ValueError("Invalid value for rep_vec. Please choose a valid option.")
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).to(device) 
        text_prompts[obj] = text_features

    return text_prompts