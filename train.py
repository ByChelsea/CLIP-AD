import torch
import numpy as np
import os
import random
import json
import torch.nn.functional as F
import argparse
from tqdm import tqdm

import open_clip
from data.dataset import get_dataset
from util.logs import init_logger
from util.losses import FocalLoss, BinaryDiceLoss
from models.model import LinearLayer
from valid import valid
from util.prompt_ensemble import encode_text_with_prompt_ensemble

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    ### init
    img_size = args.img_size
    batch_size = args.batch_size
    train_dataset = args.train_dataset
    test_dataset = args.test_dataset
    train_dataset_dir = args.train_dataset_dir
    test_dataset_dir = args.test_dataset_dir
    save_path = args.save_path
    epochs = args.epochs
    rep_vec = args.rep_vec
    features_list = args.features_list
    os.makedirs(save_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = init_logger(save_path)
    logger.info({arg: getattr(args, arg) for arg in vars(args)})
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)

    # model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)
    linear_layers = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                                len(features_list) if features_list is not None else None).to(device)
    # dataset
    train_dataloader, train_obj_list = get_dataset(train_dataset, train_dataset_dir, preprocess, img_size, batch_size)
    test_dataloader, test_obj_list = get_dataset(test_dataset, test_dataset_dir, preprocess, img_size, batch_size)
    # text prompts
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_prompts = encode_text_with_prompt_ensemble(model, train_obj_list, tokenizer, device, train_dataset, rep_vec)
    # train
    optimizer = torch.optim.Adam(list(linear_layers.parameters()), lr=args.learning_rate)
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    # training
    for epoch in range(epochs):
        loss_list = []
        for items in train_dataloader:
            cls_name = items['cls_name']
            # image
            image = items['img'].to(device)
            batch_size = image.shape[0]
            # gt
            gt_mask = items['img_mask'].squeeze().to(device)
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

            with torch.cuda.amp.autocast():
                # image features
                with torch.no_grad():
                    image_features = model.encode_image(image, features_list)  # [1, 197, 512]
                image_features = linear_layers(image_features['ori'])  # 去掉了cls token
                image_features = [feature / feature.norm(dim=-1, keepdim=True) for feature in image_features]

                # text features
                text_features = []
                for cls in cls_name:
                    text_features.append(text_prompts[cls])
                text_features = torch.stack(text_features, dim=0)  # [1, 512, 2]

                # anomaly segmentation
                anomaly_maps = []
                for idx in range(len(image_features)):
                    anomaly_map = (100.0 * image_features[idx] @ text_features) # 8, 196, 2
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, C, H, H),
                                                size=img_size, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)
                    anomaly_maps.append(anomaly_map)

            loss = 0
            for num in range(len(anomaly_maps)):
                loss += loss_focal(anomaly_maps[num], gt_mask)
                loss += loss_dice(anomaly_maps[num][:, 1, :, :], gt_mask)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        # valid
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({'linear_layers': linear_layers.state_dict()}, ckp_path)
            valid(model, linear_layers, tokenizer, test_dataloader, test_obj_list, img_size, args.visualization,
                  logger, save_path, features_list, device, test_dataset, rep_vec, args.smoothing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("SDP+ train", add_help=True)
    ## data
    parser.add_argument("--train_dataset_dir", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--test_dataset_dir", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--train_dataset", type=str, default="mvtec", help="model used")
    parser.add_argument("--test_dataset", type=str, default="mvtec", help="model used")
    parser.add_argument("--save_path", type=str, default='./results', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16.json', help="model configs")
    ## model
    parser.add_argument("--smoothing", type=int, default=8, help="Smoothing the anomaly map.")
    parser.add_argument("--rep_vec", type=str, default='mean', help='Computational methods for representative vectors.')
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--img_size", type=int, default=224, help="image size")
    parser.add_argument("--features_list", type=int, nargs="+", default=None, help="features used")
    ## train
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--visualization", action="store_true", help="Flag parameter")
    parser.add_argument("--epochs", type=int, default=1, help="batch size")
    parser.add_argument("--print_freq", type=int, default=1, help="batch size")
    parser.add_argument("--save_freq", type=int, default=1, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="batch size")
    args = parser.parse_args()

    setup_seed(42)
    train(args)

