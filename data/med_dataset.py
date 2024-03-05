import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
import csv
import cv2


class HeadCT(data.Dataset):

    def __init__(self, root, transform):
        super(HeadCT, self).__init__()
        self.root = root
        self.transform = transform
        csv_path = os.path.join(root, 'labels.csv')
        self.data_dict = {'id': [], 'hemorrhage': []}
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.data_dict['id'].append(int(row[0]))
                self.data_dict['hemorrhage'].append(int(row[1]))

    def __getitem__(self, index):
        img_id, label = self.data_dict['id'][index], self.data_dict['hemorrhage'][index]
        filename = '{:03d}.png'.format(img_id)
        img_path = os.path.join(self.root, 'head_ct', 'head_ct', filename)
        img = Image.open(img_path)
        img = self.transform(img)

        return {'img': img, 'anomaly': label, 'cls_name': 'head ct',
                'img_path': img_path}

    def __len__(self):
        return len(self.data_dict['id'])

    def get_cls_names(self):
        return ['head ct']


class BrainMRI(data.Dataset):

    def __init__(self, root, transform):
        super(BrainMRI, self).__init__()
        self.root = root
        self.transform = transform
        test_path = os.path.join(root, 'brain_tumor_dataset')
        self.image_data = []
        for class_label in ['yes', 'no']:
            if class_label == 'yes':
                label = 1
            else:
                label = 0
            class_folder = os.path.join(test_path, class_label)
            for filename in os.listdir(class_folder):
                image_path = os.path.join(class_folder, filename)
                self.image_data.append({'img_path': image_path, 'label': label})

    def __getitem__(self, index):
        img_path, label = self.image_data[index]['img_path'], self.image_data[index]['label']
        img = Image.open(img_path)
        img = self.transform(img)

        return {'img': img, 'anomaly': label, 'cls_name': 'brain ct',
                'img_path': img_path}

    def __len__(self):
        return len(self.image_data)

    def get_cls_names(self):
        return ['brain ct']

class ISIC2016(data.Dataset):

    def __init__(self, root, transform, target_transform):
        super(ISIC2016, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        img_paths = os.path.join(root, 'ISBI2016_ISIC_Part1_Test_Data')
        img_mask_paths = os.path.join(root, 'ISBI2016_ISIC_Part1_Test_GroundTruth')
        self.image_data = []
        for filename in os.listdir(img_paths):
            img_path = os.path.join(img_paths, filename)
            img_mask_path = os.path.join(img_mask_paths, filename.split('.')[0] + '_Segmentation.png')
            self.image_data.append({'img_path': img_path, 'img_mask_path': img_mask_path})

    def __getitem__(self, index):
        img_path, img_mask_path = self.image_data[index]['img_path'], self.image_data[index]['img_mask_path']
        img = Image.open(img_path)
        img = self.transform(img)
        img_mask = np.array(Image.open(img_mask_path).convert('L')) > 0
        img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        img_mask = self.target_transform(img_mask)

        return {'img': img, 'img_mask': img_mask, 'cls_name': 'skin', 'img_path': img_path}

    def __len__(self):
        return len(self.image_data)

    def get_cls_names(self):
        return ['skin']


class ClinicDB(data.Dataset):

    def __init__(self, root, transform, target_transform):
        super(ClinicDB, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        img_paths = os.path.join(root, 'Original')
        img_mask_paths = os.path.join(root, 'Ground Truth')
        self.image_data = []
        for filename in os.listdir(img_paths):
            img_path = os.path.join(img_paths, filename)
            img_mask_path = os.path.join(img_mask_paths, filename)
            self.image_data.append({'img_path': img_path, 'img_mask_path': img_mask_path})

    def __getitem__(self, index):
        img_path, img_mask_path = self.image_data[index]['img_path'], self.image_data[index]['img_mask_path']
        img = cv2.imread(img_path, 1)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transform(img)
        img_mask = cv2.imread(img_mask_path)[:, :, 0]
        img_mask = Image.fromarray(img_mask.astype(np.uint8), mode='L')
        img_mask = self.target_transform(img_mask)

        return {'img': img, 'img_mask': img_mask, 'cls_name': 'colon', 'img_path': img_path}

    def __len__(self):
        return len(self.image_data)

    def get_cls_names(self):
        return ['colon']
