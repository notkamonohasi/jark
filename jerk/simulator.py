
from typing import Union
import math

from logger import EpisodeLogger, TotalLogger
from vehicle import Vehicle
from signals import Signal, Aspect
from intersection import Intersection
from lane import Lane
from DQN.DQN import DQN
from util import calculate_euclidean_distance


BASIC_DISTANCE = 150


class Simulator : 
    def __init__(self, init_data : dict[str, any], total_logger : TotalLogger, dqn : DQN) : 
        self.delta_t = init_data["delta_t"]
        self.pos_episode = init_data["pos_episode"]
        self.step_count = 0 
        
        self.simulation_end_flag = False

        # loggerを初期化
        self.episode_logger = EpisodeLogger({
            "log_interval" : init_data["log_interval"], 
            "episode_path" : init_data["episode_path"], 
            "pos_episode" : init_data["pos_episode"]
        })

        # vehicleを初期化
        vehicle_init_data_list : list[dict[str, any]] = init_data["vehicle_init_data_list"]
        self.vehicle_dict : dict[int, Vehicle] = {
            vehicle_init_data["number"] : Vehicle(vehicle_init_data, self) for vehicle_init_data in vehicle_init_data_list
        }

        # signalを初期化
        signal_init_data_list : list[dict[str, any]] = init_data["signal_init_data_list"]
        self.signal_dict : dict[int, Signal] = {
            signal_init_data["number"] : Signal(signal_init_data, self) for signal_init_data in signal_init_data_list
        }

        # intersectionを初期化
        intersection_init_data_list : list[dict[str, any]] = init_data["intersection_init_data_list"]
        self.intersection_dict : dict[int, Intersection] = {
            intersection_init_data["number"] : Intersection(intersection_init_data, self) for intersection_init_data in intersection_init_data_list
        }

        # laneを初期化
        lane_init_data_list : list[dict[str, any]] = init_data["lane_init_data_list"]
        self.lane_dict : dict[int, Lane] = {
            lane_init_data["number"] : Lane(lane_init_data, self) for lane_init_data in lane_init_data_list
        }
        for lane in self.lane_dict.values() : 
            lane.update(self.vehicle_dict)

        # 設定
        self.limit_velocity = init_data["limit_velocity"]
        self.limit_accel = init_data["limit_accel"]
        self.limit_brake = init_data["limit_brake"]
        self.limit_step_count = init_data["limit_step_count"]

        self.total_logger = total_logger
        self.dqn = dqn


    def start(self) -> None : 
        while self.simulation_end_flag == False : 
            self.increment() 
        self.episode_logger.write_log()


    def increment(self) -> None : 
        # 各vehicleが時刻tの状況を認識（内部の状態は変化しない）
        for vehicle in self.vehicle_dict.values() : 
            vehicle.recognize() 

        # 各vehicleが意思決定（更新はまだしない）
        for vehicle in self.vehicle_dict.values() : 
            vehicle.decide_action() 

        # 各vehicleの状態を更新する
        for vehicle in self.vehicle_dict.values() : 
            vehicle.update() 

        # 各信号の状態を更新する
        for signal in self.signal_dict.values() : 
            signal.update()

        # 各laneの状態を更新する
        for lane in self.lane_dict.values() : 
            lane.update(self.vehicle_dict)

        # ステップ数を更新
        # 以降の処理では時刻がずれていることに注意する
        self.step_count += 1

        # 各vehicleが時刻t+1の状況を認識
        for vehicle in self.vehicle_dict.values() : 
            vehicle.recognize() 

        # シミュレーションを終了するかの判断
        self.judge_simulation_end()

        # 各vehicleが経験を格納
        for vehicle in self.vehicle_dict.values() : 
            vehicle.push_experience()

        # NNを更新
        self.dqn.optimize()


    def judge_simulation_end(self) -> None : 
        # 時間がかかりすぎた場合強制終了
        over_limit_step_count = False
        if self.step_count >= self.limit_step_count : 
            over_limit_step_count = True
            reason = "time over"

        # 一つの車がゴールしたら終了
        one_vehicle_goal = False 
        for vehicle in self.vehicle_dict.values() : 
            one_vehicle_goal = one_vehicle_goal or vehicle.is_goal
        if one_vehicle_goal : 
            reason = "one vehicle goal"

        # 全ての車がゴールしたら終了
        all_vehicle_goal = True
        for vehicle in self.vehicle_dict.values() : 
            all_vehicle_goal = all_vehicle_goal and vehicle.is_goal
        if all_vehicle_goal : 
            reason = "all vehicle goal"

        # 衝突している車があったら終了
        collision = False 
        for vehicle in self.vehicle_dict.values() : 
            # vehicleが発生したターンはstateが存在しないのでtry-catchする
            try : 
                if vehicle.state["front_vehicle_distance"] < 0 : 
                    collision = True
                    reason = "collision"
            except : 
                pass

        self.simulation_end_flag = over_limit_step_count or one_vehicle_goal or all_vehicle_goal or collision
        if self.simulation_end_flag : 
            print(reason)
            for vehicle in self.vehicle_dict.values() : 
                vehicle.is_goal = True


    def get_lane_length(self, lane_index) -> int : 
        return self.lane_dict[lane_index].length
    

    def get_second(self) -> float : 
        return self.delta_t * self.step_count
    

    def get_intersection_distance(self, inter_number_1, inter_number_2) -> int : 
        place_1 = self.intersection_dict[inter_number_1].get_place()
        place_2 = self.intersection_dict[inter_number_2].get_place()
        return calculate_euclidean_distance(place_1, place_2)
    
    
    def get_front_vehicle_info(self, vehicle : Vehicle) -> dict[str, any] : 
        vehicle = self.vehicle_dict[vehicle.number]

        # このターンにゴールしている可能性あり
        if vehicle.is_goal : 
            return None
        
        # まずは今のレーンを見る
        front_vehicle_number = self.lane_dict[vehicle.lane_number].get_front_vehicle_number(vehicle.number)
        
        # 同じレーンに前の車が存在する場合はそのまま距離を計算
        # 見つからなかった場合は、見つかるまでの次のレーンを見に行く
        if front_vehicle_number != None : 
            front_vehicle_distance = self.vehicle_dict[vehicle.number].get_distance_next_intersection() - \
                                        self.vehicle_dict[front_vehicle_number].get_distance_next_intersection()
        else : 
            front_vehicle_distance = vehicle.get_distance_next_intersection()
            futere_route_list = vehicle.get_future_route_list()
            for lane_number in futere_route_list : 
                front_vehicle_number = self.lane_dict[lane_number].get_back_vehicle_number()
                if front_vehicle_number != None : 
                    front_vehicle_distance += self.vehicle_dict[front_vehicle_number].get_distance_prev_intersection()
                    break
                else : 
                    front_vehicle_distance += self.lane_dict[lane_number].length
        
        if front_vehicle_number == None : 
            return None
        else : 
            # front_vehicle_distanceを車長分修正
            # これにより、距離が負になる（＝衝突する）こともある
            front_vehicle_distance -= self.vehicle_dict[vehicle.number].length / 2 + \
                                      self.vehicle_dict[front_vehicle_number].length / 2
            return {
                "distance" : front_vehicle_distance,   # この値は車長を引いていることに注意
                "velocity" : self.vehicle_dict[front_vehicle_number].velocity, 
                "accel" : self.vehicle_dict[front_vehicle_number].accel
            }
    

    def calculate_minimum_velocity(self, distance_intersection) : 
        return self.limit_velocity * distance_intersection / BASIC_DISTANCE


    def calculate_reward(self, vehicle : Vehicle, state_t0 : dict[str, any], state_t1 : dict[str, any]) -> float : 
        if vehicle.is_goal or vehicle.decide_action_way == "IDM" : 
            return 0
        
        reward = 0
        EPSILON = 0.00001

        # TTC
        relative_velocity = state_t1["velocity"] - state_t1["front_vehicle_velocity"]
        if abs(relative_velocity) > EPSILON : 
            TTC = state_t1["front_vehicle_distance"] / relative_velocity
        else : 
            TTC = state_t1["front_vehicle_distance"] / EPSILON
        if 0 <= TTC and TTC <= 4 : 
            reward += 10 * math.log(TTC / 4)   # 右辺は正

        # efficiency
        if state_t1["velocity"] >= EPSILON : 
            headway = state_t1["front_vehicle_distance"] / state_t1["velocity"]
        else : 
            headway = state_t1["front_vehicle_distance"] / EPSILON
        # reward += self.log_normal_distribution(headway, 0.4226, 0.4365)
        reward -= ((headway - 1.26) ** 2) / 10

        reward = max(-1000, reward)   # 停止されると壊れるため

        # comfort
        # 一度無視
    

        return reward

    
    def _calculate_reward(self, vehicle : Vehicle, state_t0 : dict[str, any], state_t1 : dict[str, any]) -> float : 
        reward = 0
        
        # 衝突
        reward -= state_t1["is_collision"] * 1000
        
        if vehicle.is_goal : 
            return reward

        # 速度ボーナス
        # reward += state_t1["velocity"] ** self.delta_t
        
        # 速度制限
        reward -= (state_t1["over_velocity"] * ((state_t1["velocity"] - self.limit_velocity) ** 2)) / 10

        # 車間距離
        if state_t1["exist_front_vehicle"] : 
            reward -= min(((state_t1["proper_front_vehicle_distance"] - state_t1["front_vehicle_distance"]) ** 2) / 50, 100)
            reward -= 1.0 / state_t1["front_vehicle_distance"]

        # 速度制御
        """
        distance_intersection = state_t1["distance_intersection"]
        minimum_velocity = self.calculate_minimum_velocity(distance_intersection)
        velocity = state_t1["velocity"]
        if distance_intersection > BASIC_DISTANCE or state_t1["aspect"] == Aspect.BLUE : 
            reward -= ((velocity - self.limit_velocity) ** 2) / 20
        else : 
            if velocity > self.limit_velocity : 
                reward -= ((velocity - self.limit_velocity) ** 2) / 20
            elif velocity < minimum_velocity : 
                reward -= ((minimum_velocity - velocity) ** 2) / 20

        # 信号無視
        reward -= state_t1["ignore_signal"] * 300

        # 停止
        if state_t1["is_stop"] and state_t1["aspect"] in [Aspect.BLUE] : 
            reward -= 100
        """

        # ターン毎の減点
        # reward -= 0.5

        return reward
    

    def log_normal_distribution(self, x, mu, sigma):
        sigma_2 = sigma ** 2
        r = math.log(x) - mu
        return (1.0 / (x * math.sqrt(2.0 * math.pi * sigma_2))) * math.exp(-0.5 * (r * r) / sigma_2)

