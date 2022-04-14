import torch
import torch.nn as nn


class Flatten(nn.Module):
	def forward(self, x):
		batch_size = x.shape[0]
		return x.view(batch_size, -1)


class SpecCNN(nn.Module):
	def __init__(self, classCount):
		super(SpecCNN, self).__init__()
		self.cnn = nn.Sequential(
			nn.Conv2d(1, 8, 5, stride=1),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(8),
			nn.ReLU(),

			nn.Conv2d(8, 16, 4, stride=1),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),

			nn.Conv2d(16, 32, 5, stride=1),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 32, 2, stride=1),
			nn.MaxPool2d(kernel_size=2, stride=3),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Dropout(p=0.1),

			Flatten(),
			nn.Linear(in_features=448, out_features=classCount)
		)

	def forward(self, x, ex_features=None):
		return self.cnn(x)