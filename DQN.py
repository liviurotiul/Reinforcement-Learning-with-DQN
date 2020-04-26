import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module):

	def __init__(self, outputs, inputs):
		super(DQN, self).__init__()
		self.pool = nn.MaxPool2d(2, stride=2)
		self.pool2 = nn.MaxPool2d(3, stride=2)
		self.conv0 = nn.Conv2d(inputs, 32, kernel_size=5, stride=1) #32
		self.bn0 = nn.BatchNorm2d(32)
		self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1) #32
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2) #64
		self.bn2 = nn.BatchNorm2d(128)
		# self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) #64
		# self.bn3 = nn.BatchNorm2d(64)
		self.lin1 = nn.Linear(512, 256) #512
		self.lin2 = nn.Linear(256, outputs)

	def forward(self, x):
		x = F.relu(self.bn0(self.pool2(self.conv0(x))))
		x = F.relu(self.bn1(self.pool(self.conv1(x))))
		x = F.relu(self.bn2(self.pool(self.conv2(x))))
		x = x.view(x.size(0), -1)
		x = F.relu(self.lin2(self.lin1(x)))
		return x


class TestDQN(nn.Module):
	def __init__(self, outputs, inputs):
		super(TestDQN, self).__init__()
		self.lin1 = nn.Linear(inputs, 64)
		self.bn = nn.BatchNorm1d(64)
		self.lin2 = nn.Linear(64, 16)
		self.lin3 = nn.Linear(16, outputs)


	def forward(self, x):
		x = F.relu(self.bn(self.lin1(x)))
		x = F.relu(self.lin2(x))
		x = self.lin3(x)
		return x