import os
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import open_clip
from data.dataset import get_dataset
from util.logs import init_logger
from util.prompt_ensemble import encode_text_with_prompt_ensemble
from util.metrics import calculate_metrics, visualization
from util.surgery import clip_feature_surgery, get_similarity_map


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

    # clip
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)
    # dataset
    dataloader, obj_list = get_dataset(dataset_name, dataset_dir, preprocess, img_size, args.batch_size, obj_name=args.obj_name)
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
                results['gt_px'].append(gt_mask[idx])  # (224, 224)
        if 'anomaly' in items:
            results['gt_sp'].extend(items['anomaly'].tolist())

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image, features_list)  # [1, 197, 512]
            image_features = [feature / feature.norm(dim=-1, keepdim=True) for feature in image_features['surgery']]

            # text
            text_features = []
            for cls in cls_name:
                text_features.append(text_prompts[cls])
            text_features = torch.stack(text_features, dim=0)  # [1, 512, 2]

            # anomaly segmentation
            similarity = clip_feature_surgery(image_features, text_features)
            similarity_map = get_similarity_map(similarity[:, 1:, :], (img_size, img_size), False)
            anomaly_map = similarity_map[:, :, :, 1].cpu().numpy()
            for idx in range(batch_size):
                anomaly_map[idx] = gaussian_filter(anomaly_map[idx], sigma=args.smoothing)
                results['pr_px'].append(anomaly_map[idx])
            # anomaly classification
            text_probs = (100.0 * image_features[-1] @ text_features)[:, 0, :].softmax(dim=-1)[:, 1]
            for idx in range(batch_size):
                results['pr_sp0'].append(text_probs[idx].cpu().numpy())
                results['pr_sp1'].append(np.max(anomaly_map[idx]))

    # metrics
    calculate_metrics(results, logger, obj_list)
    if args.visualization:
        visualization(results, obj_list, img_size, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("SDP", add_help=True)
    # paths
    parser.add_argument("--dataset", type=str, default='visa', help="Name of the testing dataset, optional: mvtec, visa")
    parser.add_argument("--data_path", type=str, default="./data/visa", help="Path for the testing dataset")
    parser.add_argument("--save_path", type=str, default='./results/SDP', help='Path for storing results')
    # model
    parser.add_argument("--rep_vec", type=str, default='mean', help='Computational methods for representative vectors.')
    parser.add_argument("--model", type=str, default="ViT-L-14", help="Name of the model")
    parser.add_argument("--pretrained", type=str, default="laion400m_e31", help="Name of the pretrained weight")
    parser.add_argument("--features_list", type=int, nargs="+", default=None,
                        help="Image features used for calculating the anomaly maps")
    parser.add_argument("--obj_name", type=str, default=None, help="Objects to be detected")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--smoothing", type=int, default=8, help="Smoothing the anomaly map.")
    parser.add_argument("--visualization", action="store_true", help="Visualize results")
    args = parser.parse_args()

    test(args)
