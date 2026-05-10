import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import copy
import csv
import torch.nn as nn
import torch.nn.functional as F
import SAN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_ACTOR_LR = 3e-4
DEFAULT_CRITIC_LR = 3e-4
DEFAULT_PLIF_PROXY_LR_RATIO = 0.1
DEFAULT_PLIF_LR = 1e-4


STBP_TRACE_COLUMNS = [
    "train_it",
    "actor_update",
    "actor_loss",
    "neuron",
    "layer",
    "step",
    "batch_size",
    "pre_spike_rate",
    "post_spike_rate",
    "current_abs_mean",
    "current_abs_max",
    "volt_mean",
    "volt_std",
    "current_grad_l2",
    "current_grad_abs_mean",
    "current_grad_abs_max",
    "weight_grad_t_l2",
    "weight_grad_t_abs_mean",
    "weight_grad_t_abs_max",
    "bias_grad_t_l2",
    "bias_grad_t_abs_mean",
    "bias_grad_t_abs_max",
    "param_weight_grad_l2",
    "param_weight_grad_abs_mean",
    "param_weight_grad_abs_max",
    "param_bias_grad_l2",
    "param_bias_grad_abs_mean",
    "param_bias_grad_abs_max",
]


class STBPTraceMixin:
    def _init_stbp_trace(self, trace_stbp="No", trace_stbp_freq=100):
        self.trace_stbp = trace_stbp == "Yes"
        self.trace_stbp_freq = max(1, int(trace_stbp_freq))
        self.trace_stbp_path = os.path.join("logs", "stbp_trace", "stbp_trace.csv")
        self.actor_update_it = 0
        if self.trace_stbp:
            os.makedirs(os.path.dirname(self.trace_stbp_path), exist_ok=True)

    def _begin_stbp_trace(self):
        if not self.trace_stbp:
            return False
        if self.actor_update_it % self.trace_stbp_freq != 0:
            return False
        if not isinstance(self.actor, SAN.SNN_Actor):
            return False
        if not isinstance(self.actor.snn, SAN.SpikeMLP):
            return False
        self.actor.begin_stbp_trace({
            "train_it": self.total_it,
            "actor_update": self.actor_update_it,
        })
        return True

    @staticmethod
    def _grad_stats(prefix, grad):
        if grad is None:
            return {}
        grad = grad.detach()
        return {
            f"{prefix}_l2": grad.norm().item(),
            f"{prefix}_abs_mean": grad.abs().mean().item(),
            f"{prefix}_abs_max": grad.abs().max().item(),
        }

    def _actor_snn_layers(self):
        if not isinstance(self.actor, SAN.SNN_Actor):
            return {}
        if not isinstance(self.actor.snn, SAN.SpikeMLP):
            return {}

        layers = {
            f"hidden{idx}": layer
            for idx, layer in enumerate(self.actor.snn.hidden_layers)
        }
        layers["output"] = self.actor.snn.out_pop_layer
        return layers

    def _collect_param_grad_stats(self):
        stats_by_layer = {}
        for layer_name, layer in self._actor_snn_layers().items():
            layer_stats = {}
            layer_stats.update(self._grad_stats("param_weight_grad", layer.weight.grad))
            layer_stats.update(self._grad_stats("param_bias_grad", layer.bias.grad if layer.bias is not None else None))
            stats_by_layer[layer_name] = layer_stats
        return stats_by_layer

    def _finish_stbp_trace(self, trace_active, actor_loss):
        if not trace_active:
            return
        records = self.actor.end_stbp_trace()
        if not records:
            return
        actor_loss_value = actor_loss.detach().cpu().item()
        param_grad_stats = self._collect_param_grad_stats()
        write_header = not os.path.exists(self.trace_stbp_path) or os.path.getsize(self.trace_stbp_path) == 0
        with open(self.trace_stbp_path, "a", newline="") as trace_file:
            writer = csv.DictWriter(trace_file, fieldnames=STBP_TRACE_COLUMNS)
            if write_header:
                writer.writeheader()
            for record in records:
                record = dict(record)
                record["actor_loss"] = actor_loss_value
                record.update(param_grad_stats.get(record["layer"], {}))
                writer.writerow({column: record.get(column, "") for column in STBP_TRACE_COLUMNS})
        print(
            f"[stbp-trace] train_it={self.total_it} actor_update={self.actor_update_it} "
            f"rows={len(records)} file={self.trace_stbp_path}"
        )


class Proxy_target(nn.Module):
    """ Proxy target network """
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes):
        super(Proxy_target, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(STBPTraceMixin):
    """ TD3 algorithm without the proxy target """
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            spiking_neurons,
            plif_lr=None,
            trace_stbp="No",
            trace_stbp_freq=100,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        if plif_lr is None:
            plif_lr = DEFAULT_PLIF_LR

        if spiking_neurons == 'ANN':
            self.actor = SAN.ANN_Actor(state_dim, action_dim, max_action).to(device)
        else:
            self.actor = SAN.SNN_Actor(state_dim, action_dim, max_action, spiking_neurons).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        if spiking_neurons == "PLIF" and isinstance(self.actor, SAN.SNN_Actor) and isinstance(self.actor.snn, SAN.SpikeMLP):
            plif_params = list(self.actor.snn.plifnodes.parameters())
            plif_param_ids = {id(param) for param in plif_params}
            main_params = [param for param in self.actor.parameters() if id(param) not in plif_param_ids]
            self.actor_optimizer = torch.optim.Adam(
                [
                    {"params": main_params, "lr": DEFAULT_ACTOR_LR},
                    {"params": plif_params, "lr": plif_lr},
                ]
            )
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=DEFAULT_ACTOR_LR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=DEFAULT_CRITIC_LR)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.neurons = spiking_neurons
        self.total_it = 0
        self._init_stbp_trace(trace_stbp, trace_stbp_freq)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            self.actor_update_it += 1
            trace_active = self._begin_stbp_trace()
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self._finish_stbp_trace(trace_active, actor_loss)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)



class PT_TD3(STBPTraceMixin):
    """ TD3 algorithm with the proxy target """
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            spiking_neurons,
            hidden_sizes,
            proxy_lr,
            proxy_iters,
            plif_lr=None,
            trace_stbp="No",
            trace_stbp_freq=100,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        if plif_lr is None:
            plif_lr = proxy_lr * DEFAULT_PLIF_PROXY_LR_RATIO

        if spiking_neurons == 'ANN':
            self.actor = SAN.ANN_Actor(state_dim, action_dim, max_action).to(device)
        else:
            self.actor = SAN.SNN_Actor(state_dim, action_dim, max_action, spiking_neurons).to(device)
        self.actor_target = Proxy_target(state_dim, action_dim, max_action,hidden_sizes).to(device)
        if spiking_neurons == "PLIF" and isinstance(self.actor, SAN.SNN_Actor) and isinstance(self.actor.snn, SAN.SpikeMLP):
            plif_params = list(self.actor.snn.plifnodes.parameters())
            plif_param_ids = {id(param) for param in plif_params}
            main_params = [param for param in self.actor.parameters() if id(param) not in plif_param_ids]
            self.actor_optimizer = torch.optim.Adam(
                [
                    {"params": main_params, "lr": DEFAULT_ACTOR_LR},
                    {"params": plif_params, "lr": plif_lr},
                ]
            )
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=DEFAULT_ACTOR_LR)
        self.target_optimizer = torch.optim.Adam(self.actor_target.parameters(), lr=proxy_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=DEFAULT_CRITIC_LR)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.proxy_iters = proxy_iters
        self.neurons = spiking_neurons
        self.total_it = 0
        self._init_stbp_trace(trace_stbp, trace_stbp_freq)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        target_loss=0
        for _ in range(self.proxy_iters):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            with torch.no_grad():
                actor_action = self.actor(state)
            loss = F.mse_loss(actor_action, self.actor_target(state)).mean()
            self.target_optimizer.zero_grad()
            loss.backward()
            self.target_optimizer.step()
            target_loss += loss
        target_loss /= self.proxy_iters


        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            self.actor_update_it += 1
            trace_active = self._begin_stbp_trace()
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self._finish_stbp_trace(trace_active, actor_loss)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_target.state_dict(),filename+'_target')

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
