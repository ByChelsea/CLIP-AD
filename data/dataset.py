import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from .med_dataset import HeadCT, BrainMRI, ISIC2016, ClinicDB


def get_dataset(name, dataset_dir, preprocess, img_size, batch_size, obj_name=None, shuffle=False):
	transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.CenterCrop(img_size),
									transforms.ToTensor()])
	if name == 'visa':
		data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform, obj_name=obj_name)
	elif name == 'mvtec':
		data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform, obj_name=obj_name)
	elif name == 'headct':
		data = HeadCT(root=dataset_dir, transform=preprocess)
	elif name == 'brainmri':
		data = BrainMRI(root=dataset_dir, transform=preprocess)
	elif name == 'isic':
		data = ISIC2016(root=dataset_dir, transform=preprocess, target_transform=transform)
	elif name == 'clinicdb':
		data = ClinicDB(root=dataset_dir, transform=preprocess, target_transform=transform)
	dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
	obj_list = data.get_cls_names()
	return dataloader, obj_list


class VisaDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		meta_info = meta_info['test']

		self.cls_names = list(meta_info.keys()) if obj_name is None else [obj_name]
		for cls_name in self.cls_names:
			self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path)}


class MVTecDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		meta_info = meta_info['test']

		self.cls_names = list(meta_info.keys()) if obj_name is None else [obj_name]
		for cls_name in self.cls_names:
			self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		# transforms
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path)}
