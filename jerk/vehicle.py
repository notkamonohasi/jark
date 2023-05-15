from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .simulator import Simulator

from typing import Union 
import copy 

from util import exit_failure
from IDM import get_jerk_by_IDM

class Vehicle : 
    def __init__(self, init_data : dict[str, Union[int, float, list[int]]], simulator : Simulator) -> None:
        # 属性
        self.number = init_data["number"]
        self.length = init_data["length"]
        self.decide_action_way = init_data["decide_action_way"]   # 加速度決定方法
        
        # 速度
        self.velocity = init_data["velocity"]   # m/s 
        self.accel = init_data["accel"]   # m/s^2 
        self.jerk = init_data["jerk"]   # m/s^3 

        # 位置
        self.lane_number = init_data["lane_number"] 
        self.lane_place = init_data["lane_place"]

        # 経路
        self.route_list : list[int] = init_data["route_list"]

        # jerk
        self.jerk_cand : list[int] = init_data["jerk_cand"]

        # 設定
        self.limit_velocity = init_data["limit_velocity"]
        self.limit_accel = init_data["limit_accel"]
        self.limit_brake = init_data["limit_brake"]
        
        self.simulator : Simulator = simulator

        self.state_record = {}   # 各時刻での状態を記録

        self.is_goal = False
        self.route_index = 0   # route_listにおける何番目か route_list[route_index] == lane_numberが成立
 
    
    # 現在の状態を認識
    # next_recognizeがTrueのとき、時刻t+1の状況を認識している
    def recognize(self, next_recognize : bool) -> None : 
        # state_recordに記録があったら、計算時間の節約のためにそれを使う
        if next_recognize == False and self.simulator.step_count in self.state_record.keys() : 
            self.state = self.state_record[self.simulator.step_count]
            return

        # 自身の状態
        self.state = {
            "accel" : self.accel, 
            "velocity" : self.velocity, 
            "over_velocity" : self.velocity > self.limit_velocity, 
            "over_accel" : self.accel > self.limit_accel, 
            "over_brake" : self.accel < self.limit_brake, 
            "distance_intersection" : self.get_distance_next_intersection(), 
            "is_stop" : self.velocity < 0.01, 
            "is_goal" : self.is_goal
        }

        # 前の車の情報
        front_vehicle_info = self.simulator.get_front_vehicle_info(self)
        if front_vehicle_info == None : 
            self.state["exist_front_vehicle"] = False 
            self.state["front_vehicle_distance"] = 1000
            self.state["front_vehicle_velocity"] = self.limit_velocity
            self.state["front_vehicle_accel"] = 0
            self.state["is_collision"] = False
        else : 
            self.state["exist_front_vehicle"] = True 
            self.state["front_vehicle_distance"] = front_vehicle_info["distance"]
            self.state["front_vehicle_velocity"] = front_vehicle_info["velocity"]
            self.state["front_vehicle_accel"] = front_vehicle_info["accel"]
            self.state["is_collision"] = (front_vehicle_info["distance"] < 0)

        # 記録結果をstate_recordにpush
        if next_recognize == True : 
            self.state_record[self.simulator.step_count + 1] = copy.deepcopy(self.state)
        else : 
            self.state_record[self.simulator.step_count + 0] = copy.deepcopy(self.state)

    
    # jerk決定
    def decide_action(self) -> None : 
        if self.decide_action_way == "DQN" : 
            self.action = self.simulator.dqn.decide_action(self.state)
            self.jerk = self.jerk_cand[self.action]
        elif self.decide_action_way == "IDM" : 
            self.jerk = get_jerk_by_IDM(self)
        else : 
            exit_failure("invalid Vehicle::decide_action")

    
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
        self.accel += self.jerk * self.simulator.delta_t 
        self.velocity += self.accel * self.simulator.delta_t
        self.velocity = max(0, self.velocity)   # 速度が負になるのを防ぐ


    def update_place(self) : 
        travel = self.prev_velocity * self.simulator.delta_t + 0.5 * self.accel * (self.simulator.delta_t ** 2)
        travel = max(travel, 0)   # 速度が0のとき、travelが負になる
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
        state      = self.state_record[self.simulator.step_count]
        next_state = self.state_record[self.simulator.step_count + 1]
        reward = self.simulator.calculate_reward(state, next_state)

        if self.decide_action_way == "DQN" : 
            self.simulator.dqn.push_experience(state, self.action, next_state, reward, self.is_goal)


    # 次の交差点までの距離を取得する
    def get_distance_next_intersection(self) -> float : 
        # すでにゴールしている時もある
        if self.is_goal : 
            return 0 
        else : 
            return self.simulator.get_lane_length(self.lane_number) - self.lane_place
        

    # 前の交差点からの距離を取得する
    def get_distance_prev_intersection(self) -> float : 
        # この関数は既にゴールしているときには呼び出されないはず
        if self.is_goal : 
            exit_failure("already goal in Vehicle::get_distance_prev_intersection")

        return self.lane_place
        
    
    # これから通るレーン番号を返す
    def get_future_route_list(self) : 
        return self.route_list[self.route_index + 1 : ]


    def make_log(self) -> dict[str, any] : 
        return {
            "velocity" : round(self.state["velocity"], 2), 
            "accel" : round(self.state["accel"], 2), 
            "jerk" : round(self.jerk, 2), 
            "exist_front_vehicle" : self.state["exist_front_vehicle"], 
            "front_vehicle_velocity" : round(self.state["front_vehicle_velocity"], 2),
            "front_vehicle_accel" : round(self.state["front_vehicle_accel"], 2),
            "front_vehicle_distance" : round(self.state["front_vehicle_distance"], 2),
            "distance_intersection" : round(self.get_distance_next_intersection(), 2), 
            "lane_number" : self.lane_number, 
            "lane_place" : round(self.lane_place, 2)
        }