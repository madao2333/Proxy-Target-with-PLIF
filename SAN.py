import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import copy
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import cast
import utils

'''Implementation of Proxy Target Framework '''


# Parameters for the SAN
ENCODER_REGULAR_VTH = 0.999
SPIKE_PSEUDO_GRAD_WINDOW = 0.5
# Parameters for spiking neurons
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
THETA_v = -0.172
THETA_u = 0.529
THETA_r = 0.021
THETA_s = 0.132


class PLIFNode(nn.Module):
	def __init__(self, init_tau=1 / NEURON_VDECAY):
		super().__init__()
		init_w = - math.log(init_tau - 1)
		self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))

	def forward(self, current, volt, spike):
		volt = volt * self.w.sigmoid() * (1.0 - spike) + current
		return volt
	def tau(self):
		return self.w.data.sigmoid().item()

class PseudoEncoderSpikeRegular(torch.autograd.Function):
    """ Pseudo-gradient function for encoder """
    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_REGULAR_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class PopSpikeEncoder(nn.Module):
    """ Learnable Population Coding Spike Encoder with Regular Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply
        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def spike_fn(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, PseudoEncoderSpikeRegular.apply(input_tensor))

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Regular Spike Trains
        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.spike_fn(pop_volt)
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_REGULAR_VTH
        return pop_spikes


class PopSpikeDecoder(nn.Module):
    """ Population Coding Spike Decoder """
    def __init__(self, act_dim, pop_dim, output_activation=nn.Tanh):
        """
        :param act_dim: action dimension
        :param pop_dim:  population dimension
        :param output_activation: activation function added on output
        """
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, pop_act):
        """
        :param pop_act: output population activity
        :return: raw_act
        """
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        raw_act = self.output_activation(self.decoder(pop_act).view(-1, self.act_dim))
        return raw_act


class PseudoSpikeRect(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()

class atan(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.save_for_backward(x)
		ctx.alpha = alpha
		return (x > NEURON_VTH).float()

	@staticmethod
	def backward(ctx, grad_output):
		x, = ctx.saved_tensors
		grad_x = ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * (x - NEURON_VTH)).pow(2)) * grad_output
		return grad_x, None

class SpikeMLP(nn.Module):
    """ Spike MLP for LIF and CLIF with Input and Output population neurons """
    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device, neurons):
        """
        :param in_pop_dim: input population dimension
        :param out_pop_dim: output population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoSpikeRect.apply
        # Define Layers (Hidden Layers + Output Population)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0])])
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.extend([nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer])])
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim)
        self.neurons = neurons
        # One PLIF node per hidden layer and one for output population layer.
        self.plifnodes = nn.ModuleList([PLIFNode() for _ in range(self.hidden_num + 1)])

    def spike_fn(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, PseudoSpikeRect.apply(input_tensor))

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike, plifnode: PLIFNode | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        if self.neurons == 'LIF':
            current = syn_func(pre_layer_output)
            volt = volt * NEURON_VDECAY * (1. - spike) + current
            spike = self.spike_fn(volt)
        elif self.neurons == 'CLIF':
            current = current * NEURON_CDECAY + syn_func(pre_layer_output)
            volt = volt * NEURON_VDECAY * (1. - spike) + current
            spike = self.spike_fn(volt)
        elif self.neurons == "PLIF":
            if plifnode is None:
                raise ValueError('plifnode is required when neurons == "PLIF"')
            current = syn_func(pre_layer_output)
            volt = plifnode(current, volt, spike)
            spike = self.spike_fn(volt)
        else:
            raise ValueError('Neuron model can only be LIF, CLIF, PLIF, DN, and ANN')
        return current, volt, spike


    def forward(self, in_pop_spikes, batch_size):
        """
        :param in_pop_spikes: input population spikes
        :param batch_size: batch size
        :return: out_pop_act
        """
        # Define LIF Neuron states: Current, Voltage, and Spike
        hidden_states = []
        for layer in range(self.hidden_num):
            hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                  for _ in range(3)])
        out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device)
                          for _ in range(3)]
        out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)
        # Start Spike Timestep Iteration
        for step in range(self.spike_ts):
            in_pop_spike_t = in_pop_spikes[:, :, step]
            hidden_states[0][0], hidden_states[0][1], hidden_states[0][2] = self.neuron_model(
                self.hidden_layers[0], in_pop_spike_t,
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2],
                cast(PLIFNode, self.plifnodes[0]) if self.neurons == 'PLIF' else None
            )
            if self.hidden_num > 1:
                for layer in range(1, self.hidden_num):
                    hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2] = self.neuron_model(
                        self.hidden_layers[layer], hidden_states[layer-1][2],
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2],
                        cast(PLIFNode, self.plifnodes[layer]) if self.neurons == 'PLIF' else None
                    )
            out_pop_states[0], out_pop_states[1], out_pop_states[2] = self.neuron_model(
                self.out_pop_layer, hidden_states[-1][2],
                out_pop_states[0], out_pop_states[1], out_pop_states[2],
                cast(PLIFNode, self.plifnodes[-1]) if self.neurons == 'PLIF' else None
            )
            out_pop_act += out_pop_states[2]
        out_pop_act = out_pop_act / self.spike_ts
        return out_pop_act

class DynamicMLP(nn.Module):
    """ Spike MLP for DN with Input and Output population neurons """
    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device):
        """
        :param in_pop_dim: input population dimension
        :param out_pop_dim: output population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoSpikeRect.apply
        # Define Layers (Hidden Layers + Output Population)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0])])
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.extend([nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer])])
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim)

    def spike_fn(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, PseudoSpikeRect.apply(input_tensor))

    def neuron_model(self, syn_func, pre_layer_output, current, volt, u, spike) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param u: resistance of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * (1. - spike) + THETA_r * spike
        u = u + spike * THETA_s
        dv = torch.square(volt)-volt-u+current
        du = THETA_v * volt + THETA_u * u
        volt = volt + dv
        u = u + du

        spike = self.spike_fn(volt)
        return current, volt, u, spike

    def forward(self, in_pop_spikes, batch_size):
        """
        :param in_pop_spikes: input population spikes
        :param batch_size: batch size
        :return: out_pop_act
        """
        # Define LIF Neuron states: Current, Voltage, and Spike
        hidden_states = []
        for layer in range(self.hidden_num):
            hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                  for _ in range(4)])
        out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device)
                          for _ in range(4)]
        out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)
        # Start Spike Timestep Iteration
        for step in range(self.spike_ts):
            in_pop_spike_t = in_pop_spikes[:, :, step]
            hidden_states[0][0], hidden_states[0][1], hidden_states[0][2], hidden_states[0][3] = self.neuron_model(
                self.hidden_layers[0], in_pop_spike_t,
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2], hidden_states[0][3]
            )
            if self.hidden_num > 1:
                for layer in range(1, self.hidden_num):
                    hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2], hidden_states[layer][3] = self.neuron_model(
                        self.hidden_layers[layer], hidden_states[layer-1][3],
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2], hidden_states[layer][3]
                    )
            out_pop_states[0], out_pop_states[1], out_pop_states[2], out_pop_states[3] = self.neuron_model(
                self.out_pop_layer, hidden_states[-1][3],
                out_pop_states[0], out_pop_states[1], out_pop_states[2], out_pop_states[3]
            )
            out_pop_act += out_pop_states[3]
        out_pop_act = out_pop_act / self.spike_ts
        return out_pop_act

class SNN_Actor(nn.Module):
    """ Population Coding Spike Actor with Fix Encoder """
    def __init__(self, obs_dim, act_dim, act_limit, neurons, en_pop_dim=10, de_pop_dim=10, hidden_sizes=[256,256],
                 mean_range=(-1,1), std=math.sqrt(0.05), spike_ts=5, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param en_pop_dim: encoder population dimension
        :param de_pop_dim: decoder population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param spike_ts: spike timesteps
        :param act_limit: action limit
        :param device: device
        :param use_poisson: if true use Poisson spikes for encoder
        """
        super().__init__()
        self.act_limit = act_limit
        self.encoder = PopSpikeEncoder(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        if neurons == 'DN':
            self.snn = DynamicMLP(obs_dim * en_pop_dim, act_dim * de_pop_dim, hidden_sizes, spike_ts, device)
        else:
            self.snn = SpikeMLP(obs_dim*en_pop_dim, act_dim*de_pop_dim, hidden_sizes, spike_ts, device, neurons)
        self.decoder = PopSpikeDecoder(act_dim, de_pop_dim)


    def forward(self, obs):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: action scale with action limit
        """
        batch_size=obs.size()[0]
        in_pop_spikes = self.encoder(torch.tanh(obs), batch_size)
        out_pop_activity = self.snn(in_pop_spikes, batch_size)
        action = self.act_limit * self.decoder(out_pop_activity)
        return action


class ANN_Actor(nn.Module):
    """ Actor network with ANN """
    def __init__(self, state_dim, action_dim, max_action):
        super(ANN_Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))