
import torch, random
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from typing import Tuple
from pathlib import Path

from .network import DQN_Network
from .memory import Memory
from .util import Transition, device

if device == torch.device("cuda") :
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class DQN : 
    def __init__(self, init_data : dict[str, any]) : 
        self.state_dimension = len(init_data["state_columns"])
        self.action_dimension = len(init_data["jerk_cand"])

        self.max_episode = init_data["max_episode"]
        self.jerk_cand = init_data["jerk_cand"]
        self.network = DQN_Network(self.state_dimension, self.action_dimension, False, 0)
        self.target_network = DQN_Network(self.state_dimension, self.action_dimension, True, init_data["target_learning_rate"])
        self.state_columns = init_data["state_columns"]
        self.batch_size = init_data["batch_size"]
        self.gamma = init_data["gamma"]
        self.model_path : Path = init_data["model_path"]

        self.memory = Memory({
            "buffer_size" : init_data["buffer_size"]
        })

        # target_networkの初期値をnetworkと一致させる
        self.target_network.inititalize_target(self.network)

        self.optimizer = optim.Adam(self.network.parameters(), lr=init_data["learning_rate"], amsgrad=True)

        self.pos_episode = 1

        self.loss_list = []


    def optimize(self) : 
        if self.batch_size * 10 >= len(self.memory) : 
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

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.loss_list.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.optimizer.step()

        # targetを更新
        self.target_network.update_target(self.network)

    # epsilon-greedy
    def decide_action(self, state : dict[str, any]) -> int : 
        if random.uniform(0, 1) <= self.calculate_epsilon() : 
            return random.randint(0, 1000) % self.action_dimension
        else : 
            with torch.no_grad() : 
                state_tensor = tensor([state[col] for col in self.state_columns], device=device, dtype=torch.float32)
                action = torch.argmax(self.network.forward(state_tensor))
                return action.item()
        

    def calculate_epsilon(self) : 
        half_episode = self.max_episode // 2
        if self.pos_episode < half_episode : 
            epsilon = (half_episode - self.pos_episode) / half_episode * 0.2 + 0.1 
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

    
    def write_result(self) -> None : 
        # DQNを保存
        torch.save(self.network, self.model_path.joinpath("model_weight.pth"))

        # lossのグラフを描画
        normalized_loss_list = self.get_normalize_list(self.loss_list)
        time_list = [i for i in range(len(normalized_loss_list))]
        plt.plot(time_list, normalized_loss_list)
        plt.xlabel("opt times", fontsize = 14)
        plt.ylabel("loss", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.yscale("log")
        plt.grid()
        plt.savefig(self.model_path.joinpath("loss.png"), bbox_inches="tight")
        plt.clf()

    
    # 移動平均を計算
    def get_normalize_list(self, target_list : list[float], size = 1000) -> list[float] : 
        ret = []
        pos_sum = sum(target_list[: size])
        for i in range(len(target_list) - size) : 
            pos_sum = pos_sum - target_list[i] + target_list[i + size]
            ret.append(pos_sum / size)
        return ret
    


