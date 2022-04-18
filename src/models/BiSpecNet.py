import torch
import torch.nn as nn
from torch.nn import functional as F


def bin_act(x):
	bin_act = torch.sign(x).detach()
	out = torch.tanh(x)
	return bin_act + out - out.detach()


class BinActivation(nn.Module):
	def __init__(self):
		super(BinActivation, self).__init__()

	def forward(self, x):
		out = bin_act(x)
		return out


class Flatten(nn.Module):
	def forward(self, x):
		batch_size = x.shape[0]
		return x.view(batch_size, -1)


class BinConv2d(nn.Conv2d):
	def __init__(self, *kargs, **kwargs):
		super(BinConv2d, self).__init__(*kargs, **kwargs)
		self.sign = BinActivation()

	def get_weight(self):
		return self.sign(self.weight)

	def forward(self, x):
		return F.conv2d(x, self.sign(self.weight), stride=self.stride, padding=self.padding)


class BinaryLinear(nn.Linear):
	def __init__(self, *kargs, **kwargs):
		super(BinaryLinear, self).__init__(*kargs, **kwargs)
		self.sign = BinActivation()

	def get_weight(self):
		return self.sign(self.weight)

	def forward(self, x):
		return F.linear(x, self.sign(self.weight))


class BinSpecCNN(nn.Module):
	def __init__(self, classCount):
		super(BinSpecCNN, self).__init__()
		self.BinCnn = nn.Sequential(
			BinConv2d(1, 8, 11, stride=1),
			nn.MaxPool2d(kernel_size=3, stride=1),
			nn.BatchNorm2d(8),
			BinActivation(),

			BinConv2d(8, 16, 7, stride=1),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(16),
			BinActivation(),

			BinConv2d(16, 32, 5, stride=1),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(32),
			BinActivation(),

			BinConv2d(32, 32, 5, stride=1),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(32),
			BinActivation(),

			BinConv2d(32, 32, 5, stride=1),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(32),
			BinActivation(),

			BinConv2d(32, 64, 3, stride=1),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(64),
			BinActivation(),

			Flatten(),
			BinaryLinear(in_features=256, out_features=classCount)
		)

	def forward(self, x):
		return self.BinCnn(x)
