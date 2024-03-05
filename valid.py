import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import open_clip
from util.metrics import calculate_metrics, visualization, normalize
from util.prompt_ensemble import encode_text_with_prompt_ensemble
from util.surgery import clip_feature_surgery, get_similarity_map

def valid(model, linear_layers, tokenizer, dataloader, obj_list, img_size, vis, logger, save_path, features_list,
          device, test_dataset, rep_vec, smoothing):
    linear_layers.eval()
    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device, test_dataset, rep_vec)
    results = {}
    results['cls_names'] = []
    results['img_path'] = []
    results['gt_sp'] = []
    results['pr_sp0'] = []
    results['pr_sp1'] = []
    # results['pr_sp'] = []
    results['gt_px'] = []
    results['pr_px'] = []
    for items in tqdm(dataloader, desc="Processing"):
        cls_name = items['cls_name']
        results['cls_names'].extend(cls_name)
        results['img_path'].extend(items['img_path'])
        image = items['img'].to(device)
        batch_size = image.shape[0]
        ### gt
        gt_mask = items['img_mask'].squeeze(1).numpy()
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        for idx in range(batch_size):
            results['gt_px'].append(gt_mask[idx])  # (224, 224)
        results['gt_sp'].extend(items['anomaly'].tolist())

        with torch.cuda.amp.autocast(), torch.no_grad():
            # image features
            image_features = model.encode_image(image, features_list)  # [1, 197, 512]

            # text features
            text_features = []
            for cls in cls_name:
                text_features.append(text_prompts[cls])
            text_features = torch.stack(text_features, dim=0)  # [1, 512, 2]

            # with fine-tuning
            image_features_surgery = image_features['surgery']
            image_features_ori = image_features['ori']
            image_features = linear_layers(image_features_ori)
            image_features = [feature / feature.norm(dim=-1, keepdim=True) for feature in image_features]
            anomaly_maps = []
            for idx in range(len(image_features)):
                anomaly_map = (100.0 * image_features[idx] @ text_features) # 8, 196, 2
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, C, H, H),
                                            size=img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            anomaly_map_ft = np.sum(anomaly_maps, axis=0)
            # without fine-tuning
            image_features = [feature / feature.norm(dim=-1, keepdim=True) for feature in image_features_surgery]
            similarity = clip_feature_surgery(image_features, text_features)
            similarity_map = get_similarity_map(similarity[:, 1:, :], (img_size, img_size), False)
            anomaly_map = similarity_map[:, :, :, 1].cpu().numpy()
            # save results
            anomaly_map += anomaly_map_ft
            for idx in range(batch_size):
                anomaly_map[idx] = gaussian_filter(anomaly_map[idx], sigma=smoothing)
                results['pr_px'].append(anomaly_map[idx])
            # anomaly classification
            image_features_surgery = [feature / feature.norm(dim=-1, keepdim=True) for feature in image_features_surgery]
            text_probs = (100.0 * image_features_surgery[-1] @ text_features)[:, 0, :].softmax(dim=-1)[:, 1]
            for idx in range(batch_size):
                results['pr_sp0'].append(text_probs[idx].cpu().numpy())
                results['pr_sp1'].append(np.max(anomaly_map[idx]))
                # results['pr_sp'].append(text_probs[idx].cpu().numpy())

    # metrics
    calculate_metrics(results, logger, obj_list)
    if vis:
        visualization(results, obj_list, img_size, save_path)
    linear_layers.train()