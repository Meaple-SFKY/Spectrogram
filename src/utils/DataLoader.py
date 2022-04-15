import gc
import os
import cv2 as cv
import numpy as np

import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def img_to_tensor(classCount, split_ratio, random_seed):
	x = []
	y = []
	classes = []
	except_image = []
	image_path = "../image"
	for root, dirs, classNames in os.walk(image_path):
		dirs.sort()
		for dir_ in dirs[0:classCount]:
			classes.append(os.path.basename(dir_))

	for index, className in enumerate(classes):
		class_path = os.path.join(image_path, className)
		for root, _, fileNames in os.walk(class_path):
			for fileName in fileNames:
				if fileName != '.DS_Store':
					pic = cv.imread(os.path.join(root, fileName), cv.IMREAD_GRAYSCALE)
					dataItem = cv.resize(pic, (201, 201), interpolation=cv.INTER_CUBIC)
					transf = transforms.ToTensor()
					imageTensor = transf(dataItem)
					# if imageTensor.size()[2] != 81:
					# 	except_image.append(fileName)
					# 	continue
					x.append(imageTensor)
					y.append(index)
	
	y = torch.from_numpy((np.array(y))).long()

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=random_seed)
	del x, y
	gc.collect()

	x_train = torch.stack(x_train, dim=0)
	x_test = torch.stack(x_test, dim=0)
	return (classes, x_train, x_test, y_train, y_test)


class TrainDatasets(Dataset):
	def __init__(self, x_train, y_train):
		self.len = x_train.shape[0]
		self.X_train = x_train
		self.Y_train = y_train

	def __getitem__(self, index):
		return self.X_train[index], self.Y_train[index]

	def __len__(self):
		return self.len


class TestDatasets(Dataset):
	def __init__(self, x_test, y_test):
		self.len = x_test.shape[0]
		self.X_test = x_test
		self.Y_test = y_test

	def __getitem__(self, index):
		return self.X_test[index], self.Y_test[index]

	def __len__(self):
		return self.len


class Loader():
	def __init__(self, classCount=5, split_ratio=0.7, random_seed=32, batch_size=20):
		self.classes, self.x_train, self.x_test, self.y_train, self.y_test = img_to_tensor(classCount, split_ratio, random_seed)
		self.batch_size = batch_size
		self.train_dataset = TrainDatasets(self.x_train, self.y_train)
		self.test_dataset = TestDatasets(self.x_test, self.y_test)

	def loader(self):
		train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
		test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
		return self.classes, train_loader, test_loader
