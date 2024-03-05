### Train on visa, validate on mvtec.
CUDA_VISIBLE_DEVICES=0 python3 train.py --save_path ./results/SDP_plus/train/train_on_visa \
--train_dataset visa --train_dataset_dir ./data/visa \
--test_dataset mvtec --test_dataset_dir ./data/mvtec \
--model ViT-B-16-plus-240 --pretrained laion400m_e31 --config_path ./open_clip/model_configs/ViT-B-16-plus-240.json \
--img_size 240 --batch_size 8 --features_list 3 6 9 12 --rep_vec dbscan \
--print_freq 1 --save_freq 1 --epochs 5 --learning_rate 0.0001

### Train on mvtec, validate on visa.
CUDA_VISIBLE_DEVICES=0 python3 train.py --save_path ./results/SDP_plus/train/train_on_mvtec \
--train_dataset mvtec --train_dataset_dir ./data/mvtec \
--test_dataset visa --test_dataset_dir ./data/visa \
--model ViT-B-16-plus-240 --pretrained laion400m_e31 --config_path ./open_clip/model_configs/ViT-B-16-plus-240.json \
--img_size 240 --batch_size 8 --features_list 3 6 9 12 --rep_vec dbscan --smoothing 2 \
--print_freq 1 --save_freq 1 --epochs 5 --learning_rate 0.0001
