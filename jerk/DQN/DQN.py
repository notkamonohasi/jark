
import torch, random
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import numpy as np 
from typing import Tuple

from .network import DQN_Network
from .memory import Memory
from .util import Transition, device

random.seed = 0
if device == torch.device("cuda") :
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class DQN : 
    def __init__(self, init_data : dict[str, any]) : 
        self.state_dimension = len(init_data["state_columns"])
        self.action_dimension = len(init_data["jark_cand"])

        self.max_episode = init_data["max_episode"]
        self.jark_cand = init_data["jark_cand"]
        self.network = DQN_Network(self.state_dimension, self.action_dimension, False, 0)
        self.target_network = DQN_Network(self.state_dimension, self.action_dimension, True, init_data["target_learning_rate"])
        self.state_columns = init_data["state_columns"]
        self.batch_size = init_data["batch_size"]
        self.gamma = init_data["gamma"]

        self.memory = Memory({
            "buffer_size" : init_data["buffer_size"]
        })

        # target_networkの初期値をnetworkと一致させる
        self.target_network.inititalize_target(self.network)

        self.optimizer = optim.Adam(self.network.parameters(), lr=init_data["learning_rate"], amsgrad=True)

        self.pos_episode = 1


    def optimize(self) : 
        if self.batch_size >= len(self.memory) : 
            return 
         
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, 
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.network.forward(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network.forward(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.optimizer.step()

        # targetを更新
        self.target_network.update_target(self.network)

    # epsilon-greedy
    def decide_action(self, state : dict[str, any]) -> int : 
        if random.uniform(0, 1) <= self.calculate_epsilon() : 
            return random.randint(0, 1000) % self.action_dimension
        else : 
            state_tensor = tensor([state[col] for col in self.state_columns], device=device, dtype=torch.float32)
            action = torch.argmax(self.network.forward(state_tensor))
            return action.item()
    

    def calculate_epsilon(self) : 
        half_episode = self.max_episode // 2
        if self.pos_episode < half_episode : 
            epsilon = (half_episode - self.pos_episode) / half_episode * 0.4 + 0.1 
        else : 
            epsilon = (self.max_episode - self.pos_episode) / half_episode * 0.1
        return epsilon
    

    def push_experience(self, state : dict[str, any], action, next_state : dict[str, any], reward, is_goal) : 
        # next_stateのみ特別扱い
        if is_goal : 
            next_state = None 
        else : 
            next_state = tensor([[next_state[col] for col in self.state_columns]], device=device)
        
        self.memory.push(
            tensor([[state[col] for col in self.state_columns]], device=device), 
            tensor([[action]], device=device), 
            next_state, 
            tensor([reward], device=device)
        )

    def get_eval_state(self) -> None : 
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, 
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.network.forward(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network.forward(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch


    


