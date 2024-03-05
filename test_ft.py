import os
import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

import open_clip
from data.dataset import get_dataset
from util.logs import init_logger
from util.prompt_ensemble import encode_text_with_prompt_ensemble
from util.metrics import calculate_metrics, visualization
from util.surgery import clip_feature_surgery, get_similarity_map
from scipy.ndimage import gaussian_filter
from models.model import LinearLayer


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    rep_vec = args.rep_vec
    os.makedirs(save_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = init_logger(save_path)
    logger.info({arg: getattr(args, arg) for arg in vars(args)})
    with open(args.config_path, 'r') as f:
        configs = json.load(f)

    # clip
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)
    linear_layers = LinearLayer(configs['vision_cfg']['width'], configs['embed_dim'], len(features_list)).to(device)
    checkpoint = torch.load(args.checkpoint)
    linear_layers.load_state_dict(checkpoint["linear_layers"])
    # dataset
    dataloader, obj_list = get_dataset(dataset_name, dataset_dir, preprocess, img_size, args.batch_size)
    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device, dataset_name, rep_vec)

    results = {'cls_names': [], 'img_path': [], 'gt_px': [], 'pr_px': [], 'gt_sp': [], 'pr_sp0': [], 'pr_sp1': []}
    for items in tqdm(dataloader, desc="Processing"):
        cls_name = items['cls_name']
        results['cls_names'].extend(cls_name)
        results['img_path'].extend(items['img_path'])
        image = items['img'].to(device)
        batch_size = image.shape[0]
        ### gt
        if 'img_mask' in items:
            gt_mask = items['img_mask'].squeeze(1).numpy()
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            for idx in range(batch_size):
                results['gt_px'].append(gt_mask[idx])
        if 'anomaly' in items:
            results['gt_sp'].extend(items['anomaly'].tolist())

        with torch.no_grad(), torch.cuda.amp.autocast():
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
                anomaly_map = (100.0 * image_features[idx] @ text_features)  # 8, 196, 2
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
                anomaly_map[idx] = gaussian_filter(anomaly_map[idx], sigma=args.smoothing)
                results['pr_px'].append(anomaly_map[idx])
            # anomaly classification
            image_features_surgery = [feature / feature.norm(dim=-1, keepdim=True) for feature in
                                      image_features_surgery]
            text_probs = (100.0 * image_features_surgery[-1] @ text_features)[:, 0, :].softmax(dim=-1)[:, 1]
            for idx in range(batch_size):
                results['pr_sp0'].append(text_probs[idx].cpu().numpy())
                results['pr_sp1'].append(np.max(anomaly_map[idx]))

    # metrics
    calculate_metrics(results, logger, obj_list)
    if args.visualization:
        visualization(results, obj_list, img_size, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("SDP+ test", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/tiaoshi', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16.json', help="model configs")
    parser.add_argument("--checkpoint", type=str, default='./ViT-B-16.json', help="model configs")
    # model
    parser.add_argument("--rep_vec", type=str, default='mean', help='Computational methods for representative vectors.')
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--smoothing", type=int, default=8, help="Smoothing the anomaly map.")
    parser.add_argument("--visualization", action="store_true", help="Flag parameter")
    args = parser.parse_args()

    test(args)
