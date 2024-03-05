### Industrial datasets
python test_ft.py --dataset mvtec --data_path ./data/mvtec \
--checkpoint ./pretrained_models/train_on_visa.pth --save_path ./results/SDP_plus/test/industrial/mvtec \
--model ViT-B-16-plus-240 --pretrained laion400m_e31 --config_path ./open_clip/model_configs/ViT-B-16-plus-240.json \
--features_list 3 6 9 12 --image_size 240 --batch_size 16 --rep_vec dbscan --visualization

python test_ft.py --dataset visa --data_path ./data/visa \
--checkpoint ./pretrained_models/train_on_mvtec.pth --save_path ./results/SDP_plus/test/industrial/visa \
--model ViT-B-16-plus-240 --pretrained laion400m_e31 --config_path ./open_clip/model_configs/ViT-B-16-plus-240.json \
--features_list 3 6 9 12 --image_size 240 --batch_size 16 --rep_vec dbscan --smoothing 2 --visualization

### Medical datasets
# Please don't forget to change the text prompts according to Sec. A in the supplementary materials.
python test_ft.py --dataset headct --data_path ./data/headct \
--checkpoint ./pretrained_models/train_on_visa.pth --save_path ./results/SDP_plus/test/medical/headct \
--model ViT-B-16-plus-240 --pretrained laion400m_e31 --config_path ./open_clip/model_configs/ViT-B-16-plus-240.json \
--features_list 3 6 9 12 --image_size 240 --batch_size 16 --rep_vec dbscan --visualization

python test_ft.py --dataset brainmri --data_path ./data/brainmri \
--checkpoint ./pretrained_models/train_on_visa.pth --save_path ./results/SDP_plus/test/medical/brainmri \
--model ViT-B-16-plus-240 --pretrained laion400m_e31 --config_path ./open_clip/model_configs/ViT-B-16-plus-240.json \
--features_list 3 6 9 12 --image_size 240 --batch_size 16 --rep_vec dbscan --visualization

python test_ft.py --dataset isic --data_path ./data/isic \
--checkpoint ./pretrained_models/train_on_mvtec.pth --save_path ./results/SDP_plus/test/medical/isic \
--model ViT-B-16-plus-240 --pretrained laion400m_e31 --config_path ./open_clip/model_configs/ViT-B-16-plus-240.json \
--features_list 3 6 9 12 --image_size 240 --batch_size 16 --rep_vec dbscan --visualization

python test_ft.py --dataset clinicdb --data_path ./data/cvc_clinicdb \
--checkpoint ./pretrained_models/train_on_mvtec.pth --save_path ./results/SDP_plus/test/medical/clinicdb \
--model ViT-B-16-plus-240 --pretrained laion400m_e31 --config_path ./open_clip/model_configs/ViT-B-16-plus-240.json \
--features_list 3 6 9 12 --image_size 240 --batch_size 16 --rep_vec dbscan --visualization
