from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .simulator import Simulator

from typing import Union 

class Vehicle : 
    def __init__(self, init_data : dict[str, Union[int, float, list[int]]], simulator : Simulator) -> None:
        # 属性
        self.number = init_data["number"]
        self.length = init_data["length"]
        
        # 速度
        self.velocity = init_data["velocity"]   # m/s 
        self.accel = init_data["accel"]   # m/s^2 
        self.jark = init_data["jark"]   # m/s^3 

        # 位置
        self.lane_number = init_data["lane_number"] 
        self.lane_place = init_data["lane_place"]

        # 経路
        self.route_list : list[int] = init_data["route_list"]

        # jark
        self.jark_cand : list[int] = init_data["jark_cand"]
        
        self.simulator : Simulator = simulator

        self.is_goal = False
        self.route_index = 0

    
    # 現在の状態を認識
    def recognize(self) -> None : 
        self.state = {
            "accel" : self.accel, 
            "velocity" : self.velocity,
            "is_goal" : self.is_goal
        }

    
    # jark決定
    def decide_action(self) -> None : 
        self.action = self.simulator.dqn.decide_action_single(self.state)
        self.jark = self.jark_cand[self.action]

    
    def update(self) -> None : 
        if self.is_goal : 
            return
        
        # 速度を更新する
        self.prev_velocity = self.velocity   # 一度速度を記録しておく必要がある
        self.update_velocity()

        # 場所を更新する
        self.update_place()

        # 更新結果を記録
        self.simulator.logger.register_vehicle_log(self.number, self.make_log())


    def update_velocity(self) : 
        self.accel += self.jark * self.simulator.delta_t 
        self.velocity += self.accel * self.simulator.delta_t


    def update_place(self) : 
        travel = self.prev_velocity * self.simulator.delta_t + 0.5 * self.accel * (self.simulator.delta_t ** 2)
        pos_lane_length = self.simulator.get_lane_length(self.lane_number)
        if self.lane_place + travel < pos_lane_length : 
            self.lane_place = self.lane_place + travel 
        else : 
            self.route_index += 1 
            if self.route_index == len(self.route_list) : 
                self.is_goal = True
                self.lane_number = -1 
                self.lane_place = 0 
            else : 
                self.lane_number = self.route_list[self.route_index]
                self.lane_place = self.lane_place + travel - pos_lane_length


    def push_experience(self) : 
        state = {key : val for key, val in self.state.items()}   # deepcopy
        self.recognize()
        next_state = self.state
        reward = self.simulator.calculate_immediate_reward(state, next_state)
        self.simulator.dqn.push_experience(state, self.action, next_state, reward, self.is_goal)


    def make_log(self) -> dict[str, any] : 
        return {
            "velocity" : round(self.velocity, 2), 
            "accel" : round(self.accel, 2), 
            "jark" : round(self.jark, 2), 
            "lane_number" : self.lane_number, 
            "lane_place" : round(self.lane_place, 2)
        }