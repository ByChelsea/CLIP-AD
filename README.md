# CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-shot Anomaly Detection

[Xuhai Chen](https://bychelsea.github.io/xuhaichen.github.io/), [Jiangning Zhang](https://zhangzjn.github.io/), Guanzhong Tian, Haoyang He, Wuhao Zhang, Yabiao Wang, Chengjie Wang, Yong Liu

This repository contains the official PyTorch implementation of the paper [CLIP-AD](https://arxiv.org/abs/2311.00453). It is an upgraded version of the method we proposed for the [competition](https://github.com/ByChelsea/VAND-APRIL-GAN).

<img src="illustration/clipad.png" alt="Qualitative comparisons" style="max-width: 50px; height: auto;">

## Installation

- Prepare experimental environments

  ```shell
  pip install -r requirements.txt
  ```
  
## Dataset Preparation 
### MVTec-AD
- Download and extract MVTec-AD into `data/mvtec`
- run`python data/mvtec.py` to obtain `data/mvtec/meta.json`
```
data
├── mvtec
    ├── meta.json
    ├── bottle
        ├── train
            ├── good
                ├── 000.png
        ├── test
            ├── good
                ├── 000.png
            ├── anomaly1
                ├── 000.png
        ├── ground_truth
            ├── anomaly1
                ├── 000.png
```

### VisA
- Download and extract VisA into `data/visa`
- run`python data/visa.py` to obtain `data/visa/meta.json`
```
data
├── visa
    ├── meta.json
    ├── candle
        ├── Data
            ├── Images
                ├── Anomaly
                    ├── 000.JPG
                ├── Normal
                    ├── 0000.JPG
            ├── Masks
                ├── Anomaly
                    ├── 000.png
```

### ISIC
- Download and extract [ISIC](https://challenge.isic-archive.com/data/) into `data/isic`
```
data
├── isic
    ├── ISBI2016_ISIC_Part1_Test_Data
        ├── ISIC_0000003.jpg
    ├── ISBI2016_ISIC_Part1_Test_GroundTruth
        ├── ISIC_0000003_Segmentation.png
```

### CVC-ClinicDB
- Download and extract [CVC-ClinicDB](https://datasetninja.com/cvc-612) into `data/cvc_clinicdb`
```
data
├── cvc_clinicdb
    ├── Ground Truth
        ├── 1.tif
    ├── Original
        ├── 1.tif
    ├── README.txt
```

### HeadCT
- Download and extract [HeadCT](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage) into `data/headct`
```
data
├── headct
    ├── head_ct
        ├── head_ct
            ├── 000.png
    ├── labels.csv
```

### BrainMRI
- Download and extract [BrainMRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) into `data/brainmri`
```
data
├── brainmri
    ├── brain_tumor_dataset
        ├── no
            ├── 1 no.jpeg
        ├── yes
            ├── Y1.jpg
    ├── no
        ├── 1 no.jpeg
    ├── yes
        ├── Y1.jpg
```

## SDP
Set parameters in `test_SDP.sh`.
- `dataset`: name of the testing dataset, optional: mvtec, visa
- `data_path`: the path to the testing dataset
- `model`: the CLIP model
- `pretrained`: the pretrained weights
- `features_list`: features of different layers to use
- `image_size`: the size of the input images
- `rep_vec`: the method for selecting representative vectors, optional: mean, pca, kde, dbscan, mean_shift

Then run the following command
  ```shell
  test_SDP.sh
  ```

## SDP+
### Training
Set parameters in `train_SDP_plus.sh`.
- `print_freq`: the frequency of printing logs
- `save_freq`: the frequency of conducting validation and saving the model
- `epochs`: total epochs

Then run the following command
  ```shell
  train_SDP_plus.sh
  ```

The pretrained models are in `./pretrained_models`.

### Testing
Set parameters in `test_SDP_plus.sh`.
- `checkpoint`: the path to the checkpoint

Then run the following command
  ```shell
  test_SDP_plus.sh
  ```

## Citation
If our work is helpful for your research, please consider citing:

```
@article{chen2023clip,
  title={Clip-ad: A language-guided staged dual-path model for zero-shot anomaly detection},
  author={Chen, Xuhai and Zhang, Jiangning and Tian, Guanzhong and He, Haoyang and Zhang, Wuhao and Wang, Yabiao and Wang, Chengjie and Wu, Yunsheng and Liu, Yong},
  journal={arXiv preprint arXiv:2311.00453},
  year={2023}
}
```
