from collections import namedtuple
import random
Experience = namedtuple('Experience',
						('states','actions','next_states','rewards'))

class ReplayMemory:
	def __init__(self,capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		
	def push(self,*args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)

		self.memory[self.position] = Experience(*args)
		self.position = int((self.position + 1)%self.capacity)
		
	def sample(self,batch_size):
		# print(len(self.memory),batch_size)
		return random.sample(self.memory,batch_size)
	
	def __len__(self):
		return len(self.memory)
	
Uncertain_Experience = namedtuple('Experience',
						('states','actions','next_states','rewards','adv_actions'))

class Uncertain_ReplayMemory:
	def __init__(self,capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		
	def push(self,*args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)

		self.memory[self.position] = Uncertain_Experience(*args)
		self.position = int((self.position + 1)%self.capacity)
		
	def sample(self,batch_size):
		# print(len(self.memory),batch_size)
		return random.sample(self.memory,batch_size)
	
	def __len__(self):
		return len(self.memory)
	
continuous_Experience = namedtuple('Experience',
						('states','actions','next_states','rewards','dt','global_state', 'next_global_state', 'ttg'))  # delta_t for continuous time step

class continuous_ReplayMemory:
	def __init__(self,capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		
	def push(self,*args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)

		self.memory[self.position] = continuous_Experience(*args)
		self.position = int((self.position + 1)%self.capacity)
		
	def sample(self,batch_size):
		# print(len(self.memory),batch_size)
		return random.sample(self.memory,batch_size)
	
	def __len__(self):
		return len(self.memory)
	

Safe_Experience = namedtuple('Experience',
						('states','actions','next_states','rewards','constraints'))

class safe_ReplayMemory:
	def __init__(self,capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		
	def push(self,*args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)

		self.memory[self.position] = Safe_Experience(*args)
		self.position = int((self.position + 1)%self.capacity)
		
	def sample(self,batch_size):
		# print(len(self.memory),batch_size)
		return random.sample(self.memory,batch_size)
	
	def __len__(self):
		return len(self.memory)
	
Epi_Experience = namedtuple('Experience',
						('states','actions','next_states','rewards','constraints', 'z', 'dts', 'ttg'))

class Epi_ReplayMemory:
	def __init__(self,capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		
	def push(self,*args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)

		self.memory[self.position] = Epi_Experience(*args)
		self.position = int((self.position + 1)%self.capacity)
		
	def sample(self,batch_size):
		# print(len(self.memory),batch_size)
		return random.sample(self.memory,batch_size)
	
	def __len__(self):
		return len(self.memory)