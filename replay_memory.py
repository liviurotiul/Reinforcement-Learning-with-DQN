import numpy as np
import torch

class ReplayMemory():
	def __init__(self, capactiy=1000, batch_size=128):
		self.capacity = capactiy
		self.memory = []
		self.position = 0
		self.batch_size = batch_size

	def push_back(self, state, action, inst_reward, next_state):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = [state, action, inst_reward, next_state]
		self.position = (self.position + 1) % self.capacity
	
	def __len__(self):
		return len(self.memory)

	def sample(self):
		samples = np.random.randint(low=0, high=len(self.memory)-1, size=self.batch_size)
		batch = np.asarray(self.memory)[samples]
		# for i in range(len(batch)):
		# 	batch[i][1] = torch.LongTensor(batch[i][1])
		# 	batch[i][2] = torch.LongTensor(batch[i][2])
		return batch.tolist()

class PrioritizedReplayMemory():
	def __init__(self, capactiy=1000, batch_size=128, ALPHA=0.5):
		self.capacity = capactiy
		self.memory = []
		self.position = 0
		self.batch_size = batch_size

	def push_back(self, state, action, inst_reward, next_state):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = [state, action, inst_reward, next_state]
		self.position = (self.position + 1) % self.capacity
	
	def __len__(self):
		return len(self.memory)

	def sample(self):
		samples = np.random.randint(low=0, high=len(self.memory)-1, size=self.batch_size)
		batch = np.asarray(self.memory)[samples].tolist()
		batch = [torch.from_numpy(item) for item in batch]
		return batch

