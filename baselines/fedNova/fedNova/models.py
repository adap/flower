'''
Modified from https://github.com/pytorch/vision.git
'''
import os.path

import torch
import math
import torch.nn as nn
import torch.nn.init as init
from torch.optim.optimizer import Optimizer, required
from utils import Meter, comp_accuracy
from typing import List, Tuple
import numpy as np


class VGG(nn.Module):
	'''
	VGG model
	'''

	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(512, 512),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(512, 512),
			nn.ReLU(True),
			nn.Linear(512, 10),
		)
		# Initialize weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				m.bias.data.zero_()

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


cfg = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
		  512, 512, 512, 512, 'M'],
}


def vgg11():
	"""VGG 11-layer model (configuration "A")"""
	return VGG(make_layers(cfg['A']))


def train(model, optimizer, trainloader, device, epochs):
	criterion = nn.CrossEntropyLoss()
	# optimizer = ProxSGD(model.parameters(), ratio, optimizer_config['gmf'], optimizer_config['mu'],
	# 					optimizer_config['lr'], optimizer_config['momentum'], optimizer_config['dampening'], optimizer_config['weight_decay'],
	# 					optimizer_config['nesterov'], optimizer_config['variance'])

	model.train()

	for epoch in range(epochs):
		losses = Meter(ptag='Loss')
		top1 = Meter(ptag='Prec@1')
		for batch_idx, (data, target) in enumerate(trainloader):
			# data loading
			data = data.to(device)
			target = target.to(device)

			# forward pass
			output = model(data)
			loss = criterion(output, target)

			# backward pass
			loss.backward()

			# gradient step
			optimizer.step()
			optimizer.zero_grad()

			# write log files
			train_acc = comp_accuracy(output, target)

			losses.update(loss.item(), data.size(0))
			top1.update(train_acc[0].item(), data.size(0))

			# print("Epoch: {} Batch: {} Loss: {} Acc: {}".format(epoch, batch_idx, loss.item(), train_acc[0].item()))

			# if batch_idx == 5:
			# 	break


def test(model, test_loader, device) -> Tuple[float, float]:
	criterion = nn.CrossEntropyLoss()
	model.eval()
	top1 = Meter(ptag='Acc')
	total_loss = 0.0

	with torch.no_grad():
		for data, target in test_loader:
			data = data.to(device)
			target = target.to(device)
			outputs = model(data)
			total_loss += criterion(outputs, target).item()
			acc1 = comp_accuracy(outputs, target)
			top1.update(acc1[0].item(), data.size(0))

	total_loss /= len(test_loader)
	return total_loss, top1.avg


class ProxSGD(Optimizer):
	"""
	SGD optimizer modified with support for proximal updates
	Args:
		params (iterable): iterable of parameters to optimize or dicts defining
			parameter groups
		ratio (float): relative sample size of client
		gmf (float): global/server/slow momentum factor
		mu (float): parameter for proximal local SGD
		lr (float): learning rate
		momentum (float, optional): momentum factor (default: 0)
		weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
		dampening (float, optional): dampening for momentum (default: 0)
		nesterov (bool, optional): enables Nesterov momentum (default: False)
	"""

	def __init__(self, params, ratio: float, gmf=0, mu=0, lr=required, momentum=0, dampening=0,
				 weight_decay=0, nesterov=False, variance=0):

		self.gmf = gmf
		self.ratio = ratio
		self.momentum = momentum
		self.mu = mu
		self.local_normalizing_vec = 0
		self.local_counter = 0
		self.local_steps = 0
		self.lr = lr

		if lr is not required and lr < 0.0:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if momentum < 0.0:
			raise ValueError("Invalid momentum value: {}".format(momentum))
		if weight_decay < 0.0:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

		defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
						weight_decay=weight_decay, nesterov=nesterov, variance=variance)
		if nesterov and (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum requires a momentum and zero dampening")
		super(ProxSGD, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(ProxSGD, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('nesterov', False)

	def step(self, closure=None):
		"""Performs a single optimization step.
		"""
		# scale = 1**self.itr

		for group in self.param_groups:
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']

			for p in group['params']:
				if p.grad is None:
					continue
				d_p = p.grad.data

				if weight_decay != 0:
					d_p.add_(p.data, alpha=weight_decay)

				param_state = self.state[p]
				if 'old_init' not in param_state:
					param_state['old_init'] = torch.clone(p.data).detach()

				local_lr = group['lr']

				# apply momentum updates
				if momentum != 0:
					if 'momentum_buffer' not in param_state:
						buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
					else:
						buf = param_state['momentum_buffer']
						buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
					if nesterov:
						d_p = d_p.add(momentum, buf)
					else:
						d_p = buf

				# apply proximal updates
				if self.mu != 0:
					d_p.add_(p.data - param_state['old_init'], alpha=self.mu)

				# update accumalated local updates
				if 'cum_grad' not in param_state:
					param_state['cum_grad'] = torch.clone(d_p).detach()
					param_state['cum_grad'].mul_(local_lr)

				else:
					param_state['cum_grad'].add_(d_p, alpha=local_lr)

				p.data.add_(d_p, alpha=-local_lr)

		# compute local normalizing vector a_i
		if self.momentum != 0:
			self.local_counter = self.local_counter * self.momentum + 1
			self.local_normalizing_vec += self.local_counter

		self.etamu = local_lr * self.mu
		if self.etamu != 0:
			self.local_normalizing_vec *= (1 - self.etamu)
			self.local_normalizing_vec += 1

		if self.momentum == 0 and self.etamu == 0:
			self.local_normalizing_vec += 1

		self.local_steps += 1

	def get_local_stats(self) -> List:
		if self.mu != 0:
			local_tau = torch.tensor(self.local_steps * self.ratio)
		else:
			local_tau = torch.tensor(self.local_normalizing_vec * self.ratio)
		local_stats = {"weight": self.ratio, "tau": local_tau.item(), "local_norm": self.local_normalizing_vec}

		return local_stats

	def load_aggregated_grad(self, agg_grad):
		if len(agg_grad) == 0:
			i = 0
			# print("---------------No parameters received, filling it with init params------------")
			init_params = torch.load("state/init_params_torch.pt")
			for group in self.param_groups:
				for p in group['params']:
					p.data.copy_(init_params[i])
					i+=1
			return None

		i = 0
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				param_state['cum_grad'] = torch.tensor(agg_grad[i],device=param_state["cum_grad"].device)
				i += 1

		self.reset_local_buffers()
		# print("-------------------optimizer load aggregated grad completed --------------------------")

	def reset_local_buffers(self):
		# Call this after every round of federated training
		for group in self.param_groups:
			lr = group['lr']
			for p in group['params']:
				param_state = self.state[p]

				if self.gmf != 0:
					if 'global_momentum_buffer' not in param_state:
						buf = param_state['global_momentum_buffer'] = torch.clone(param_state['cum_grad']).detach()
						buf.div_(lr)
					else:
						buf = param_state['global_momentum_buffer']
						buf.mul_(self.gmf).add_(param_state['cum_grad'], 1 / lr)
					param_state['old_init'].sub_(buf, alpha=lr)
				else:
					param_state['old_init'].sub_(param_state['cum_grad'])

				p.data.copy_(param_state['old_init'])
				param_state['cum_grad'].zero_()

				# Reinitialize momentum buffer
				if 'momentum_buffer' in param_state:
					param_state['momentum_buffer'].zero_()

		self.local_counter = 0
		self.local_normalizing_vec = 0
		self.local_steps = 0


if __name__ == "__main__":
	import ssl

	ssl._create_default_https_context = ssl._create_unverified_context

	from torch.optim import SGD
	from dataset import load_datasets
	from omegaconf import OmegaConf

	config = OmegaConf.create({"datapath": "data/", "num_clients": 1, "NIID": True, "alpha": 0.2, "batch_size": 8})

	device = torch.device('cuda')
	# torch.manual_seed(args.seed)
	# torch.cuda.manual_seed(args.seed)
	# torch.cuda.set_device('0')
	# torch.backends.cudnn.deterministic = True

	model = vgg11().to(device)
	params = model.parameters()
	# saved_params = [val.cpu().numpy() for _, val in model.state_dict().items()]
	# torch.save(saved_params, "state/init_params_torch.pt")
	# torch.save(params, "state/init_params.pt")

	optimizer = ProxSGD(params, 0.0625, lr=0.1, weight_decay=1e-4, momentum=0)
	# init_params = np.load("state/init_params.npy", allow_pickle=True)
	init_params= torch.load("state/init_params_torch.pt")
	# init_params = saved_params

	i= 0
	import time
	print("started")
	t1 = time.time()
	for group in optimizer.param_groups:
		for p in group['params']:
			# p.data.copy_(torch.from_numpy(init_params[i]).to(device))
			p.data.copy_(init_params[i])
			i+=1

	print("took  : {} seconds ".format(time.time() - t1))


	# criterion = nn.CrossEntropyLoss()

	# trainloader, testloader, ratio = load_datasets(config)
	# print(len(testloader))
	# model.train()
	# files = os.listdir("state/")
	# client = ["0", "1", "2", "3"]
	# param = {}
	# for cid in client:
	# 	param[cid] = torch.load(f"state/model_{cid}_3.pt")
	#
	# for i in range(3):
	# 	all_equal = [np.isclose(param[client[i]][j], param[client[i+1]][j]) for j in range(len(param[client[i]]))]
	# 	print([(np.sum(x), x.shape) for x in all_equal])
	# 	break

	# for epoch in range(1):
	# 	losses = Meter(ptag='Loss')
	# 	top1 = Meter(ptag='Prec@1')
	# 	for batch_idx, (data, target) in enumerate(trainloader[0]):
	# 		# data loading
	# 		data = data.to(device)
	# 		target = target.to(device)
	#
	# 		# forward pass
	# 		output = model(data)
	# 		loss = criterion(output, target)
	#
	# 		# backward pass
	# 		loss.backward()
	#
	# 		# gradient step
	# 		optimizer.step()
	# 		optimizer.zero_grad()
	#
	# 		# write log files
	# 		train_acc = comp_accuracy(output, target)
	#
	# 		losses.update(loss.item(), data.size(0))
	# 		top1.update(train_acc[0].item(), data.size(0))
	# 		print("loss:{} acc: {}".format(loss.item(), train_acc[0].item()))
	# 		break


	# params = optimizer.state_dict()
	# torch.save(params, "state/proxsgd.pt")
