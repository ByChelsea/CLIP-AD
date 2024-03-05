### Industrial datasets
python test.py --dataset mvtec --data_path ./data/mvtec \
--save_path ./results/SDP/industrial/mvtec --model ViT-B-16-plus-240 --features_list 3 6 9 12 --pretrained laion400m_e31 \
--image_size 240 --batch_size 16 --rep_vec dbscan --visualization

python test.py --dataset visa --data_path ./data/visa \
--save_path ./results/SDP/industrial/visa --model ViT-B-16-plus-240 --features_list 3 6 9 12 --pretrained laion400m_e31 \
--image_size 240 --batch_size 16 --rep_vec dbscan --visualization

### Medical datasets
# Please don't forget to change the text prompts according to Sec. A in the supplementary materials.
python test.py --dataset headct --data_path ./data/headct \
--save_path ./results/SDP/medical/headct --model ViT-B-16-plus-240 --features_list 3 6 9 12 --pretrained laion400m_e31 \
--image_size 240 --batch_size 16 --rep_vec dbscan --visualization

python test.py --dataset brainmri --data_path ./data/brainmri \
--save_path ./results/SDP/medical/brainmri --model ViT-B-16-plus-240 --features_list 3 6 9 12 --pretrained laion400m_e31 \
--image_size 240 --batch_size 16 --rep_vec dbscan --visualization

python test.py --dataset isic --data_path ./data/isic \
--save_path ./results/SDP/medical/isic --model ViT-B-16-plus-240 --features_list 3 6 9 12 --pretrained laion400m_e31 \
--image_size 240 --batch_size 16 --rep_vec dbscan --visualization

python test.py --dataset clinicdb --data_path ./data/cvc_clinicdb \
--save_path ./results/SDP/medical/clinicdb --model ViT-B-16-plus-240 --features_list 3 6 9 12 --pretrained laion400m_e31 \
--image_size 240 --batch_size 16 --rep_vec dbscan --visualization


