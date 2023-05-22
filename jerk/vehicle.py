from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .simulator import Simulator

from typing import Union 
import copy 

from util import exit_failure
from IDM import get_jerk_by_IDM
from signals import Aspect

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
        self.ignore_signal = False
        self.route_index = 0   # route_listにおける何番目か route_list[route_index] == lane_numberが成立
 
    
    # 現在の状態を認識
    def recognize(self) -> None : 
        # state_recordに記録があったら、計算時間の節約のためにそれを使う
        if self.simulator.step_count in self.state_record.keys() : 
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
            "is_goal" : self.is_goal, 
            "ignore_signal" : self.ignore_signal
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

        # 信号情報
        signal_state = self.get_front_signal_state()
        if signal_state == None : 
            self.state["BLUE"] = None
            self.state["YELLOW_TO_RED"] = None 
            self.state["RED"] = None 
            self.state["YELLOW_TO_BLUE"] = None
            self.state["remain_time"] = None
        else : 
            self.state["BLUE"] = 0
            self.state["YELLOW_TO_RED"] = 0
            self.state["RED"] = 0
            self.state["YELLOW_TO_BLUE"] = 0
            self.state[signal_state["aspect"].name] = 1
            self.state["remain_time"] = signal_state["remain_time"]

        # 記録結果をstate_recordにpush
        self.state_record[self.simulator.step_count] = copy.deepcopy(self.state)

    
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


    def update_velocity(self) : 
        self.accel += self.jerk * self.simulator.delta_t 
        self.accel = min(self.limit_accel, self.accel)   # 最大値を超えないようにする
        self.accel = max(self.limit_brake, self.accel)   # 最小値を超えないようにする
        self.velocity += self.accel * self.simulator.delta_t
        self.velocity = max(0, self.velocity)   # 速度が負になるのを防ぐ


    def update_place(self) : 
        travel = self.prev_velocity * self.simulator.delta_t + 0.5 * self.accel * (self.simulator.delta_t ** 2)
        travel = max(travel, 0)   # 速度が0のとき、travelが負になる
        pos_lane_length = self.simulator.get_lane_length(self.lane_number)
        if self.lane_place + travel < pos_lane_length :   # tとt+1で同一レーン
            self.lane_place = self.lane_place + travel 
        else :   # t+1で違うレーンに移動
            # 信号を守ったかチェック
            self.ignore_signal = False
            intersection_number = self.simulator.lane_dict[self.lane_number].to_intersection_number
            signal_number = self.simulator.intersection_dict[intersection_number].signal_number
            if signal_number == None : 
                pass 
            else : 
                signal_state = self.simulator.signal_dict[signal_number].get_signal_state()
                aspect : Aspect = signal_state["aspect"]
                if aspect in [Aspect.RED, Aspect.YELLOW_TO_BLUE] : 
                    self.ignore_signal = True

            # レーン番号とレーン位置を修正   
            self.route_index += 1 
            if self.route_index == len(self.route_list) :   # ゴール
                self.is_goal = True
                self.lane_number = -1 
                self.lane_place = 0 
            else :   # レーン移動
                self.lane_number = self.route_list[self.route_index]
                self.lane_place = self.lane_place + travel - pos_lane_length


    def push_experience(self) : 
        # このときのsimulatorの時刻はt+1であることに注意
        state      = self.state_record[self.simulator.step_count - 1]
        next_state = self.state_record[self.simulator.step_count]
        action = self.action
        reward = self.simulator.calculate_reward(state, next_state)

        if self.decide_action_way == "DQN" : 
            self.simulator.dqn.push_experience(state, action, next_state, reward, self.is_goal)

        # 経験をloggerに登録
        self.simulator.logger.register_vehicle_log(self.number, self.make_log(state, reward))


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
    

    # 次の交差点の信号の情報を取得する
    def get_front_signal_state(self) -> dict[str, Union(Aspect, float)] : 
        # このターンにゴールした
        if self.lane_number == -1 : 
            return None
        
        next_intersection_number = self.simulator.lane_dict[self.lane_number].to_intersection_number
        signal_number = self.simulator.intersection_dict[next_intersection_number].signal_number
        if signal_number == None : 
            return None 
        else : 
            return self.simulator.signal_dict[signal_number].get_signal_state()
        
    
    # これから通るレーン番号を返す
    def get_future_route_list(self) : 
        return self.route_list[self.route_index + 1 : ]


    def make_log(self, state : dict[str, any], reward) -> dict[str, any] : 
        return {
            "reward" : round(reward, 2), 
            "velocity" : round(state["velocity"], 2), 
            "accel" : round(self.state["accel"], 2), 
            "jerk" : round(self.jerk, 2), 
            "exist_front_vehicle" : round(self.state["exist_front_vehicle"], 2), 
            "front_vehicle_velocity" : round(self.state["front_vehicle_velocity"], 2),
            "front_vehicle_accel" : round(self.state["front_vehicle_accel"], 2),
            "front_vehicle_distance" : round(self.state["front_vehicle_distance"], 2),
            "distance_intersection" : round(self.get_distance_next_intersection(), 2), 
            "ignore_signal" : self.state["ignore_signal"], 
            "BLUE" : self.state["BLUE"], 
            "YELLOW_TO_RED" : self.state["YELLOW_TO_RED"], 
            "RED" : self.state["RED"], 
            "YELLOW_TO_BLUE" : self.state["YELLOW_TO_BLUE"], 
            "remain_time" : self.state["remain_time"], 
            "lane_number" : self.lane_number, 
            "lane_place" : round(self.lane_place, 2)
        }