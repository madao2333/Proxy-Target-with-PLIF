import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import copy
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, cast
import SAN
import actor_critic

'''Implementation of Proxy Target Framework '''

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


def eval_policy(policy: Union[actor_critic.TD3, actor_critic.PT_TD3], env_name, eval_seed, eval_episodes=10):
	'''
	Runs policy for X episodes and returns average reward and per-layer tau values.
	A fixed seed is used for the eval environment.
	'''
	eval_env = gym.make(env_name)
	eval_env.reset(seed=eval_seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		state = state[0]
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done1, done2, _ = eval_env.step(action)
			done = done1 + done2
			avg_reward += float(reward)

	avg_reward /= eval_episodes
	eval_taus = []
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	if policy.neurons == "PLIF":
		if isinstance(policy.actor, SAN.SNN_Actor):
			snn_module = policy.actor.snn
			if isinstance(snn_module, SAN.SpikeMLP):
				eval_taus = [cast(SAN.PLIFNode, node).tau() for node in snn_module.plifnodes]
				if len(eval_taus) > 1:
					hidden_tau_str = ", ".join(
						f"h{idx}: {tau:.6f}" for idx, tau in enumerate(eval_taus[:-1])
					)
					print(f"Current PLIF tau (hidden): {hidden_tau_str}")
					print(f"Current PLIF tau (output): {eval_taus[-1]:.6f}")
				else:
					print(f"Current PLIF tau: {eval_taus[0]:.6f}")
	return avg_reward, eval_taus

	
