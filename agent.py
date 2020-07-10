import numpy as np
import copy
import random
from collections import namedtuple, deque

import torch
import torch.optim as optim
import torch.nn.functional as F

from actor import Actor
from critic import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG():
    """DDPG agent"""
    
    def __init__(self, state_size, action_size, config, seed):
        """Initialize a DDPG agent
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            ddpg_config (config): configuration of DDPG 
            seed (int): random seed
        """
        
        self.gamma = config.gamma
        self.tau = config.tau
        self.seed = np.random.seed(seed)
        
        # actor networks
        self.actor_local = Actor(state_size, action_size, config.units_actor, seed).to(device)
        self.actor_target = Actor(state_size, action_size, config.units_actor, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), config.lr_actor)
        
        # critic newtworks
        self.critic_local = Critic(state_size, action_size, config.units_critic, seed).to(device)
        self.critic_target = Critic(state_size, action_size, config.units_critic, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), config.lr_critic)
        
        # Noise process
        self.noise = OUNoise(action_size, seed, config.mu, config.theta, config.sigma) 
        
        # Replay Buffer
        self.memory = ReplayBuffer(config.buffer_size, config.batch_size, seed)
        
    def noise_reset(self):
        self.noise.reset()
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay buffer, and use random sample from buffer to learn."""
        
        self.memory.add(state, action,  reward, next_state, done)
        
        # learn if enough sample in memory
        if len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            
    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).data.cpu().numpy()
        self.actor_local.train()
        action +=  self.noise.sample()
        
        return np.clip(action, -1, 1)
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples
        
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
            
         Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples 
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        #### Update critic 
        # Get predicted next-state actions from actor_target model
        next_actions = self.actor_target(next_states)
        
        # Get predicted next-state Q-Values from critic_target model
        next_q_targets = self.critic_target(next_states, next_actions)     
        
        # Compute Q targets for current states
        Q_targets = rewards + self.gamma * next_q_targets*(1.0-dones)
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        ### Update actor
        # Compute actor loss
        predicted_actions = self.actor_local(states)
        actor_loss = -self.critic_local(states, predicted_actions).mean()
        
        # Minimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
        ### Update target networks
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)  

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.seed = np.random.seed(seed)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
            seed (int): random seed
        """
        self.seed = np.random.seed(seed)
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        """ convert to states, actions, rewards, next_states and dones of the selected experiences to tensors """
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return  (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    