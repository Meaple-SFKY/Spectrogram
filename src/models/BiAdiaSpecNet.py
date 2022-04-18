import torch
import torch.nn as nn
from torch.nn import functional as F


def bin_act(x, w):
	out = torch.tanh(x / w)

	return out


class BinActivation(nn.Module):
	def __init__(self):
		super(BinActivation, self).__init__()
		self.w = 1
		self.mode = 0

	def set_mode(self, w, mode):
		self.w = w
		self.mode = mode

	def forward(self, x):
		if self.mode == 0:
			out = x
		elif self.mode == 1:
			out = bin_act(x, self.w)
		else:
			out = torch.sign(x)

		return out


class BinConv2d(nn.Conv2d):
	def __init__(self, *kargs, **kwargs):
		super(BinConv2d, self).__init__(*kargs, **kwargs)
		self.w = 1
		self.mode = 0
		self.sign = BinActivation()

	def set_mode(self, w, mode):
		self.w = w
		self.mode = mode
		self.sign.set_mode(self.w, self.mode)

	def get_weight(self):
		return self.sign(self.weight)

	def forward(self, x):
		return F.conv2d(x, self.sign(self.weight), stride=self.stride, padding=self.padding)


class BinaryLinear(nn.Linear):
	def __init__(self, *kargs, **kwargs):
		super(BinaryLinear, self).__init__(*kargs, **kwargs)
		self.w = 1
		self.mode = 0
		self.sign = BinActivation()

	def set_mode(self, w, mode):
		self.w = w
		self.mode = mode
		self.sign.set_mode(self.w, self.mode)

	def get_weight(self):
		return self.sign(self.weight)

	def forward(self, x):
		return F.linear(x, self.sign(self.weight))


class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)


class BiAdiaNet(nn.Module):
	def __init__(self, classCount):
		super(BiAdiaNet, self).__init__()

		self.act_w = 1
		self.wei_w = 1
		self.act_mode = 0
		self.wei_mode = 0

		self.conv01 = BinConv2d(1, 8, 11, stride=1, bias=False)
		self.conv02 = BinConv2d(8, 32, 7, stride=1, bias=False)
		self.conv03 = BinConv2d(32, 64, 5, stride=1, bias=False)
		self.conv04 = BinConv2d(64, 64, 5, stride=1, bias=False)
		self.conv05 = BinConv2d(64, 128, 5, stride=1, bias=False)
		self.conv06 = BinConv2d(128, 128, 3, stride=1, bias=False)
		self.linear = BinaryLinear(512, classCount, bias=False)
		self.pool01 = nn.MaxPool2d(3, stride=1)
		self.pool02 = nn.MaxPool2d(3, stride=2)
		self.pool03 = nn.MaxPool2d(3, stride=2)
		self.pool04 = nn.MaxPool2d(3, stride=2)
		self.pool05 = nn.MaxPool2d(3, stride=2)
		self.pool06 = nn.MaxPool2d(3, stride=2)
		self.batn01 = nn.BatchNorm2d(8)
		self.batn02 = nn.BatchNorm2d(32)
		self.batn03 = nn.BatchNorm2d(64)
		self.batn04 = nn.BatchNorm2d(64)
		self.batn05 = nn.BatchNorm2d(128)
		self.batn06 = nn.BatchNorm2d(128)
		self.actv01 = BinActivation()
		self.actv02 = BinActivation()
		self.actv03 = BinActivation()
		self.actv04 = BinActivation()
		self.actv05 = BinActivation()
		self.actv06 = BinActivation()
		self.flatten = Flatten()

	def set_act_mode(self, w, mode):
		self.act_w = w
		self.act_mode = mode
		self.actv01.set_mode(self.act_w, self.act_mode)
		self.actv02.set_mode(self.act_w, self.act_mode)
		self.actv03.set_mode(self.act_w, self.act_mode)
		self.actv04.set_mode(self.act_w, self.act_mode)
		self.actv05.set_mode(self.act_w, self.act_mode)
		self.actv06.set_mode(self.act_w, self.act_mode)

	def set_wei_mode(self, w, mode):
		self.wei_w = w
		self.wei_mode = mode
		self.conv01.set_mode(self.wei_w, self.wei_mode)
		self.conv02.set_mode(self.wei_w, self.wei_mode)
		self.conv03.set_mode(self.wei_w, self.wei_mode)
		self.conv04.set_mode(self.wei_w, self.wei_mode)
		self.conv05.set_mode(self.wei_w, self.wei_mode)
		self.conv06.set_mode(self.wei_w, self.wei_mode)
		self.linear.set_mode(self.wei_w, self.wei_mode)
	
	def get_mode(self):
		return (self.act_w, self.wei_w, self.act_mode, self.wei_mode)

	def get_weight(self):
		return (
			self.conv01.get_weight(),
			self.conv02.get_weight(),
			self.conv03.get_weight(),
			self.conv04.get_weight(),
			self.conv05.get_weight(),
			self.conv06.get_weight(),
			self.linear.get_weight()
		)

	def forward(self, x):
		x = self.actv01(self.batn01(self.pool01(self.conv01(x))))
		x = self.actv02(self.batn02(self.pool02(self.conv02(x))))
		x = self.actv03(self.batn03(self.pool03(self.conv03(x))))
		x = self.actv04(self.batn04(self.pool04(self.conv04(x))))
		x = self.actv05(self.batn05(self.pool05(self.conv05(x))))
		x = self.actv06(self.batn06(self.pool06(self.conv06(x))))
		x = self.linear(self.flatten(x))
		return x
