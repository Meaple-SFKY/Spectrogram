import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam


class OpeModel():
	def __init__(self, model, lr, trloader, teloader):
		self.model = model
		self.train_loader = trloader
		self.test_loader = teloader
		self.criterion = nn.CrossEntropyLoss()
		self.optimi = Adam(self.model.parameters(), lr=lr)
		self.acc_list, self.loss_list = [], []

	def modify_lr(self, value):
		for p in self.optimi.param_groups:
			p['lr'] = value

	def train(self):
		self.model.train()
		for data in self.train_loader:
			inputs, target = data
			self.optimi.zero_grad()
			outputs = self.model(inputs)
			loss = self.criterion(outputs, target)
			loss.backward()
			self.optimi.step()
		loss_item = loss.item()
		self.loss_list.append(loss_item)
		return loss_item

	def test(self):
		self.model.eval()
		correct, total = 0, 0
		for data in self.test_loader:
			inputs, target = data
			outputs = self.model(inputs)
			_, predicted = torch.max(outputs.data, dim=1)
			total += len(target)
			correct += (predicted == target).sum().item()
		correct_item = (100 * correct) / total
		self.acc_list.append(correct_item)
		return correct_item

	def train_strategy(self, epoch):
		for index in range(epoch):
			loss = self.train()
			acc = self.test()
			print('Loss, Acc - %d: %.8f, %.2f %%' % (index, loss, acc))

	def eval_acc(self, epoch=5):
		sum = 0.0
		for i in range(epoch):
			self.model.eval()
			correct, total = 0, 0
			for data in self.test_loader:
				inputs, target = data
				outputs = self.model(inputs)
				_, predicted = torch.max(outputs.data, dim=1)
				total += len(target)
				correct += (predicted == target).sum().item()
			sum += (100 * correct) / total
		sum /= epoch
		print('model acc: %.2f' % (sum))
		return sum

	def save_state(self, mode, end, cnt):
		acc = self.eval_acc(3)
		torch.save(self.model, '../model/%s,%.2f,%d,%d.pt' % (mode, acc, end, cnt))
		np.save('../tmp/%s,acc,%d,%d' % (mode, end, cnt), self.acc_list)
		np.save('../tmp/%s,loss,%d,%d' % (mode, end, cnt), self.loss_list)

	def load_state(self, mode, acc, end, cnt):
		self.model = torch.load('../model/%s,%.2f,%d,%d.pt' % (mode, acc, end, cnt))
		self.acc_list = np.load('../tmp/%s,acc,%d,%d.npy' % (mode, end, cnt)).tolist()
		self.loss_list = np.load('../tmp/%s,loss,%d,%d.npy' % (mode, end, cnt)).tolist()

	def get_state(self):
		return (self.model, self.acc_list, self.loss_list)
