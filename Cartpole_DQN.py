


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import time
import copy
import cv2
from utils import *
from DQN import *
from replay_memory import *
from gym import wrappers
import gc
from collections import defaultdict, OrderedDict


###################################################################################################
#	FLAGS
###################################################################################################


ALGORITHM = 0
'''
0 - vanilla DQN
1 - Double DQN
2 - vanilla DQN with PER #not implemented
3 - Double DQN with PER #not implemented
4 - test on api
'''
OPT_FREQ = 4
SHOW_IMG = False
FULL_IMG = False
RGB = False
DEBUG = False
VALIDATION_FREQ = 50
BATCH_SIZE = 42
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 7000
TARGET_UPDATE = int(1000*np.sqrt(OPT_FREQ))
EPISODES = 100000
VAL_EPISODES = 20
EVAL = True
TRAIN = True
ALPHA = np.random.randint(1)
# ENVIRIONMENT = 'LunarLander-v2'
ENVIRIONMENT = 'CartPole-v0'
TEST = False
LR = 3e-7
CAP = 31500
EPS_ADAM = 1e-3

steps_done = 0


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device != "cpu":
	print(" Will run on gpu")
else:
	print(" Will run on cpu")


#get the environment
environment = gym.make(ENVIRIONMENT).unwrapped
n_actions = environment.action_space.n



#some preproc
preprocesing = None
if ENVIRIONMENT == 'CartPole-v0':
	if not RGB:
		preprocesing = T.Compose([T.ToPILImage(),
								T.Resize(40, interpolation=Image.CUBIC),
								T.Grayscale(num_output_channels=1)])
	else:
		preprocesing = T.Compose([T.ToPILImage(),
								T.Resize(40, interpolation=Image.CUBIC)])
else:
	if not RGB:
		preprocesing = T.Compose([T.ToPILImage(),
								T.Resize((40,90), interpolation=Image.CUBIC),
								T.Grayscale(num_output_channels=1)])
	else:
		preprocesing = T.Compose([T.ToPILImage(),
								T.Resize((40,90), interpolation=Image.CUBIC)])	

preprocesing2 = T.Compose([T.ToTensor()])

#replay memory
memory = None
if ALGORITHM == 2 or ALGORITHM == 3:
	memory = PrioritizedReplayMemory(batch_size=BATCH_SIZE)
else:
	memory = ReplayMemory(capactiy=CAP, batch_size=BATCH_SIZE) #15000

if not TEST:
	if RGB and FULL_IMG:
		online_net = DQN(n_actions,6).to(device)
		target_net = DQN(n_actions,6).to(device)
		bestaverage_net = DQN(n_actions,6).to(device)
		bestval_net = DQN(n_actions,6).to(device)
	elif RGB and not FULL_IMG:
		online_net = DQN(n_actions,3).to(device)
		target_net = DQN(n_actions,3).to(device)
		bestaverage_net = DQN(n_actions,3).to(device)
		bestval_net = DQN(n_actions,3).to(device)
	elif not RGB and not FULL_IMG:
		online_net = DQN(n_actions,1).to(device)
		target_net = DQN(n_actions,1).to(device)
		bestaverage_net = DQN(n_actions,1).to(device)
		bestval_net = DQN(n_actions,1).to(device)
	elif not RGB and FULL_IMG:
		online_net = DQN(n_actions,2).to(device)
		target_net = DQN(n_actions,2).to(device)
		bestaverage_net = DQN(n_actions,2).to(device)
		bestval_net = DQN(n_actions,2).to(device)
else:
	online_net = TestDQN(n_actions,8).to(device)
	target_net = TestDQN(n_actions,8).to(device)
	bestaverage_net = TestDQN(n_actions,8).to(device)
	bestval_net = TestDQN(n_actions,8).to(device)

#models
target_net.load_state_dict(online_net.state_dict())
# optimizer1 = None
# optimizer2 = None
# optimizer = None
if ALGORITHM == 1 or ALGORITHM == 3:
	optimizer1 = optim.Adam(online_net.parameters(), lr=LR, eps=20e-3) # TODO change dampening term//RMSPROP cu decay
	optimizer2 = optim.Adam(target_net.parameters(), lr=LR, eps=20e-3)
elif ALGORITHM == 0 or ALGORITHM == 2:
	# optimizer = optim.Adam(target_net.parameters(), lr=0.003, eps=0.000001)
	optimizer = optim.Adam(online_net.parameters(), lr=LR, eps=EPS_ADAM)


def get_cart_location(screen_width):
	world_width = environment.x_threshold * 2
	scale = screen_width / world_width
	return int(environment.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen_cartpole():
	screen = environment.render(mode='rgb_array').transpose((2, 0, 1))
	_, screen_height, screen_width = screen.shape
	screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
	view_width = int(screen_width * 0.6)
	cart_location = get_cart_location(screen_width)
	if cart_location < view_width // 2:
		slice_range = slice(view_width)
	elif cart_location > (screen_width - view_width // 2):
						slice_range = slice(-view_width, None)
	else:
		slice_range = slice(cart_location - view_width // 2,
							cart_location + view_width // 2)
	screen = screen[:, :, slice_range]
	screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
	screen = torch.from_numpy(screen)
	# Resize, and add a batch dimension (BCHW)
	screen = preprocesing(screen)
	screen = T.functional.adjust_contrast(screen, 7)
	screen = preprocesing2(screen)
	return screen.unsqueeze(0).to(device)

def get_screen_lander():
	screen = environment.render(mode='rgb_array').transpose((2, 0, 1))
	_, screen_height, screen_width = screen.shape
	screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
	screen = torch.from_numpy(screen)
	# Resize, and add a batch dimension (BCHW)
	screen = preprocesing(screen)
	screen = T.functional.adjust_contrast(screen, 7)
	screen = preprocesing2(screen)
	# import pdb;pdb.set_trace()
	return screen.unsqueeze(0).to(device)

def select_action(state): #vanilla DQN
	global steps_done
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
		math.exp(-1. * steps_done / EPS_DECAY)

	if steps_done%300 == 0 and DEBUG:
		print(eps_threshold)

	steps_done += 1

	if sample > eps_threshold:
		with torch.no_grad():
			online_net.eval()
	# t.max(1) will return largest column value of each row.
	# second column on max result is index of where max element was
	# found, so we pick action with the larger expected reward.
	# print(policy_net(state), policy_net(state).max(1)[1].view(1, 1))
			return online_net(state).max(1)[1].view(1, 1)
	else:
		return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def select_action_opt(state):
	with torch.no_grad():
		return target_net(state).max(1)[1].view(1, 1)

def select_action_rand():
	return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


###################################################################################################
#	PLOTTING STUFF
###################################################################################################


episode_durations = []

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

def plot_durations():
	plt.figure(1)
	plt.clf()
	episode_durations_local = np.asarray(episode_durations).transpose()
	validation_values_local = np.asarray(validation_values).transpose()
	durations_t = torch.tensor(episode_durations_local[1], dtype=torch.float)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')

	plt.plot(episode_durations_local[0], episode_durations_local[1], label="episode rewards", color="blue")
	plt.plot(validation_values_local[0], validation_values_local[1], label="validation values(everey few episodes)", color="orange")
    # Take 100 episode averages and plot them too
	if len(durations_t) >= 100:
		means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(episode_durations_local[0], means.numpy().tolist(), label="episode duration mean(last 100)", color="green")

	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())


###################################################################################################
#	OPTIMIZE FUNTION
###################################################################################################

def optimize_DQN():
	if len(memory) < BATCH_SIZE:
		return
	online_net.train()
	# Prepare for optimization
	batch = memory.sample()
	current_states = np.asarray(batch).transpose()[0].tolist() # list with all states to enter
	nonfinal_mask = torch.tensor([mask(item) for item in np.asarray(batch).transpose()[3].tolist()]).to(device) # get the mask // 1 for non final, 0 for final
	# print(current_states)
	current_states = torch.cat(current_states, dim=0) # concatenate to send it to the target network
	actions = torch.cat((np.asarray(batch).transpose()[1]).tolist()).to(device).unsqueeze(1)

	non_final_next_states = torch.cat([s for s in np.asarray(batch).transpose()[3].tolist()
												if s is not None])
	# in the state_values will be found the values of the future states given we make actions based on policy
	# pi and coming from state S (current_states) 
	if DEBUG:
		print(current_states)
		print(target_net(current_states))
		print(nonfinal_mask)
	state_values = torch.zeros(BATCH_SIZE, device=device) # where the mask is 0 ergo the next state is final the reward will be 0
	state_values[nonfinal_mask] = target_net(non_final_next_states).max(1)[0].detach()
	inst_rewards = torch.cat(np.asarray(batch).transpose()[2].tolist()).to(device)
	expected_state_action_values = (state_values * GAMMA) + inst_rewards # This is our target tensor


	if steps_done % 100 == 0:
		print(torch.mean(state_values))

	# we are basicly comparing what the online network says our maximum reward will be choosing the 
	# actions we allready chose in the next state to what we now our instant reward is and the predicted value of
	# that next state will be by the target network
	# print(actions.size())
	state_action_values = online_net(current_states).gather(1, actions) # This is our input tensor


	# the calsical optimizer routine
	loss = F.smooth_l1_loss(state_action_values.double(), expected_state_action_values.unsqueeze(1).double())
	optimizer.zero_grad()
	loss.backward()
	# for param in online_net.parameters():
	# 	param.grad.data.clamp_(-1, 1)
	optimizer.step()
	if DEBUG:
		print("state_values:")
		print(state_values)
		print("nonfinal_mask")
		print(nonfinal_mask)
		print("action batch:")
		print(actions)
		print("rewards batch:")
		print(inst_rewards)
		print("expected_state_action_values batch:")
		print(expected_state_action_values)
		print("state_action_values batch:")
		print(state_action_values)
		input("next")
		print("---------------------------------------------------------")

def optimize_double_DQN():
	ALPHA = np.random.randn(1)

	if len(memory) < BATCH_SIZE:
		return

	# Prepare for optimization
	batch = memory.sample()
	current_states = np.asarray(batch).transpose()[0].tolist() # list with all states to enter
	nonfinal_mask = torch.tensor([mask(item) for item in np.asarray(batch).transpose()[3].tolist()]).to(device) # get the mask // 1 for non final, 0 for final
	current_states = torch.cat(current_states, dim=0) # concatenate to send it to the target network
	actions = torch.cat((np.asarray(batch).transpose()[1]).tolist()).to(device).unsqueeze(1)

	non_final_next_states = torch.cat([s for s in np.asarray(batch).transpose()[3].tolist()
												if s is not None])

	# in the state_values will be found the values of the future states given we make actions based on policy
	# pi and coming from state S (current_states) 
	if DEBUG:
		print(target_net(current_states))
		print(nonfinal_mask)
	state_values = torch.zeros(BATCH_SIZE, device=device) # where the mask is 0 ergo the next state is final the reward will be 0
	if ALPHA > 0.5:
		state_values[nonfinal_mask] = target_net(non_final_next_states).max(1)[0].detach() # TODO : fix this
	else:
		state_values[nonfinal_mask] = online_net(non_final_next_states).max(1)[0].detach()
	inst_rewards = torch.cat(np.asarray(batch).transpose()[2].tolist()).to(device)
	expected_state_action_values = (state_values * GAMMA) + inst_rewards # This is our target tensor



	# we are basicly comparing what the online network says our maximum reward will be choosing the 
	# actions we allready chose in the next state to what we now our instant reward is and the predicted value of
	# that next state will be by the target network
	# print(actions.size())
	if ALPHA > 0.5:
		state_action_values = online_net(current_states).gather(1, actions) # This is our input tensor
	else:
		state_action_values = target_net(current_states).gather(1, actions) # This is our input tensor


	# the calsical optimizer routine
	loss = F.smooth_l1_loss(state_action_values.double(), expected_state_action_values.unsqueeze(1).double())

	optimizer1.zero_grad()
	loss.backward()
	optimizer1.step()

	if DEBUG:
		print("state_values:")
		print(state_values)
		print("nonfinal_mask")
		print(nonfinal_mask)
		print("action batch:")
		print(actions)
		print("rewards batch:")
		print(inst_rewards)
		print("expected_state_action_values batch:")
		print(expected_state_action_values)
		print("state_action_values batch:")
		print(state_action_values)
		input("next")
		print("---------------------------------------------------------")

def optimize_double_DQN_PER():
	ALPHA = np.random.randn(1)

	if len(memory) < BATCH_SIZE:
		return

	# Prepare for optimization
	batch = memory.sample()
	current_states = np.asarray(batch).transpose()[0].tolist() # list with all states to enter
	nonfinal_mask = torch.tensor([mask(item) for item in np.asarray(batch).transpose()[3].tolist()]).to(device) # get the mask // 1 for non final, 0 for final
	current_states = torch.cat(current_states, dim=0) # concatenate to send it to the target network
	actions = torch.cat((np.asarray(batch).transpose()[1]).tolist()).to(device).unsqueeze(1)

	non_final_next_states = torch.cat([s for s in np.asarray(batch).transpose()[3].tolist()
												if s is not None])

	# in the state_values will be found the values of the future states given we make actions based on policy
	# pi and coming from state S (current_states) 
	if DEBUG:
		print(target_net(current_states))
		print(nonfinal_mask)
	state_values = torch.zeros(BATCH_SIZE, device=device) # where the mask is 0 ergo the next state is final the reward will be 0
	if ALPHA > 0.5:
		state_values[nonfinal_mask] = target_net(non_final_next_states).max(1)[0].detach()
	else:
		state_values[nonfinal_mask] = online_net(non_final_next_states).max(1)[0].detach()
	inst_rewards = torch.cat(np.asarray(batch).transpose()[2].tolist()).to(device)
	expected_state_action_values = (state_values * GAMMA) + inst_rewards # This is our target tensor



	# we are basicly comparing what the online network says our maximum reward will be choosing the 
	# actions we allready chose in the next state to what we now our instant reward is and the predicted value of
	# that next state will be by the target network
	# print(actions.size())
	if ALPHA > 0.5:
		state_action_values = online_net(current_states).gather(1, actions) # This is our input tensor
	else:
		state_action_values = target_net(current_states).gather(1, actions) # This is our input tensor


	# the calsical optimizer routine
	loss = F.smooth_l1_loss(state_action_values.double(), expected_state_action_values.unsqueeze(1).double())
	loss_each = F.smooth_l1_loss(state_action_values.double(), expected_state_action_values.unsqueeze(1).double())
	import pdb; pdb.set_trace()
	if ALPHA > 0.5:
		optimizer1.zero_grad()
		loss.backward()
		# for param in online_net.parameters():
		# 	param.grad.data.clamp_(-1, 1)
		optimizer1.step()
	else:
		optimizer2.zero_grad()
		loss.backward()
		# for param in online_net.parameters():
		# 	param.grad.data.clamp_(-1, 1)
		optimizer2.step()
	if DEBUG:
		print("state_values:")
		print(state_values)
		print("nonfinal_mask")
		print(nonfinal_mask)
		print("action batch:")
		print(actions)
		print("rewards batch:")
		print(inst_rewards)
		print("expected_state_action_values batch:")
		print(expected_state_action_values)
		print("state_action_values batch:")
		print(state_action_values)
		input("next")
		print("---------------------------------------------------------")

###################################################################################################
#	TRAINING LOOP
###################################################################################################

new_efficiency = old_efficiency = 0
validation_values = []
old_means = 0
get_screen = None
if ENVIRIONMENT == 'CartPole-v0':
	get_screen = get_screen_cartpole
else:
	get_screen = get_screen_lander

t = False
if TRAIN:
	for episode in range(EPISODES):


	##########################################################
	#	VALIDATION SECTION
	##########################################################
		
		if episode % VALIDATION_FREQ == 0:
			print("validating")
			target_net.eval()
			validation_rewards = []

			for episode in range(VAL_EPISODES):
				reward_ep = 0
				next_state = None
				environment.reset()

				if not TEST: # get our STATE
					last_screen = get_screen()
					current_screen = get_screen()
					if FULL_IMG:
						# import pdb; pdb.set_trace()
						current_state = torch.cat([current_screen, last_screen], 1)
					else:
						current_state = current_screen - last_screen 
				else:
					last_obs, _, _, _ = environment.step(0)
					current_obs = last_obs
					current_state = torch.cat([torch.from_numpy(current_obs), torch.from_numpy(last_obs)]).float().unsqueeze(0).to(device)
					# print(current_state)

				for time_step in count():
					# import pdb; pdb.set_trace()
					action = select_action_opt(current_state) # get our ACTION

					if not TEST:
						_, reward, done, _ = environment.step(action.item()) # get the REWARD and DONE signal
						last_screen = current_screen
						current_screen = get_screen()
						if not done:
							if FULL_IMG:
								next_state = torch.cat([current_screen, last_screen], 1)
							else:
								next_state = current_screen - last_screen
						else:	
							next_state = None
					else:
						last_obs = current_obs
						current_obs, reward, done, _ = environment.step(action.item()) # get the REWARD and DONE signal
						if not done:
							next_state = torch.cat([torch.from_numpy(current_obs), torch.from_numpy(last_obs)]).float().unsqueeze(0).to(device)
						else:
							next_state = None
					reward_ep += reward

					current_state = next_state
					if done:
						break
				validation_rewards.append(reward_ep)
			new_efficiency = sum(validation_rewards)/len(validation_rewards)
			if new_efficiency > old_efficiency:
				bestval_net.load_state_dict(target_net.state_dict())
				old_efficiency = new_efficiency
				print("best validation updated")
			print(new_efficiency)
			validation_values.append([steps_done, new_efficiency])
			if new_efficiency > 70 and t == False:
				OPT_FREQ = OPT_FREQ * 3 + 3
				TARGET_UPDATE = int(TARGET_UPDATE * 1.5)
				t = True
				print("OPFREQ Tripled")
			if new_efficiency > 500:
				input("ai atins performant gringo")


	##########################################################


		current_state = None; next_state = None
		environment.reset()
		next_state = None

		if not TEST: # get our STATE
			last_screen = get_screen()
			current_screen = get_screen()
			if FULL_IMG:
						# import pdb; pdb.set_trace()
				current_state = torch.cat([current_screen, last_screen], 1)
			else:
				current_state = current_screen - last_screen 
		else:
			last_obs, _, _, _ = environment.step(0)
			current_obs = last_obs
			current_state = torch.cat([torch.from_numpy(current_obs), torch.from_numpy(last_obs)]).float().unsqueeze(0)
			# print(current_state)


		total_reward = 0
		for time_step in count():
			online_net.eval()
			if not TEST:
				if SHOW_IMG:
					if ENVIRIONMENT == 'CartPole-v0':
						show(current_state, RGB)
					else:
						show(current_screen, RGB)
				action = select_action(current_state) # get our ACTION
				_, reward, done, _ = environment.step(action.item()) # get the REWARD and DONE signal
				total_reward += reward
				last_screen = current_screen
				current_screen = get_screen()
				if not done:
					if FULL_IMG:
						next_state = torch.cat([current_screen, last_screen], 1)
					else:
						next_state = current_screen - last_screen
				else:	
					next_state = None
			else:
				last_obs = current_obs
				action = select_action(current_state) # get our ACTION
				current_obs, reward, done, _ = environment.step(action.item()) # get the REWARD and DONE signal
				total_reward += reward

				if not done:
					next_state = torch.cat([torch.from_numpy(current_obs), torch.from_numpy(last_obs)]).float().unsqueeze(0).to(device)
				else:
					next_state = None
			memory.push_back(current_state,
							torch.from_numpy(np.asarray([action.item()])).cpu(),
							torch.from_numpy(np.asarray([reward])).cpu(),
							next_state)
			current_state = next_state
			try:
				if steps_done % OPT_FREQ == 0:
					if ALGORITHM == 1:
						optimize_double_DQN()
					elif ALGORITHM == 3:
						optimize_double_DQN_PER()
					elif ALGORITHM == 0:
						optimize_DQN()
			except:
				print("An unkown error has occurred when optimizing")
			if steps_done % TARGET_UPDATE == 0:# and ALGORITHM != 1 and ALGORITHM != 3:
				target_net.load_state_dict(online_net.state_dict())
			if DEBUG:
				count_tensors()
			if done:
				break

		episode_durations.append([steps_done, total_reward])
		if not SHOW_IMG:
			plot_durations()

		if episode > 100:
			new_means = np.sum(np.asarray(episode_durations).transpose()[1][-100:-1])
			if old_means<new_means:
				old_means = new_means
				bestaverage_net.load_state_dict(target_net.state_dict())
				print("best average updated")

	torch.save(bestaverage_net, "./models/cartpole/bestaverage_net")
	torch.save(bestval_net, "./models/cartpole/bestval_net")
input(" Finished Training")





print(" Models saved")

###################################################################################################
#	EVALUATION
###################################################################################################


if EVAL:
	models = ["./models/cartpole/bestaverage_net","./models/cartpole/bestval_net"]
	print("evaluating")
	for model_name in models:
		print("evaluating", model_name)
		model = torch.load(model_name)
		model.eval()
		validation_rewards = []

		for episode in range(30):
			reward_ep = 0
			environment.reset()
			next_state = None
			environment.reset()
			last_screen = get_screen()
			current_screen = get_screen()
			if FULL_IMG:
				# import pdb; pdb.set_trace()
				current_state = torch.cat([current_screen, last_screen], 1)
			else:
				current_state = current_screen - last_screen # get our STATE
			for time_step in count():
				action = model(current_state).max(1)[1].view(1, 1) # get our ACTION
				# action = select_action_rand()
				_, reward, done, _ = environment.step(action.item()) # get the REWARD and DONE signal
				reward_ep += reward
				last_screen = current_screen
				current_screen = get_screen()
				if not done:
					if FULL_IMG:
						next_state = torch.cat([current_screen, last_screen], 1)
					else:
						next_state = current_screen - last_screen
				else:	
					next_state = None
				current_state = next_state
				if done:
					break
			validation_rewards.append(reward_ep)
		print(sum(validation_rewards)/len(validation_rewards))
plt.plot(validation_rewards, label="rewards per episode")
plt.show()
input("daa")