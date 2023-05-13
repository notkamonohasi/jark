
from typing import Union

from logger import Logger
from vehicle import Vehicle
from lane import Lane
from DQN.DQN import DQN

class Simulator : 
    def __init__(self, init_data : dict[str, any], dqn : DQN) : 
        self.delta_t = init_data["delta_t"]
        self.step_count = 0 

        # loggerを初期化
        self.logger = Logger({
            "log_interval" : init_data["log_interval"], 
            "result_path" : init_data["result_path"], 
            "pos_episode" : init_data["pos_episode"]
        })

        # vehicleを初期化
        vehicle_init_data_list : list[dict[str, any]] = init_data["vehicle_init_data_list"]
        self.vehicle_dict : dict[int, Vehicle] = {
            vehicle_init_data["number"] : Vehicle(vehicle_init_data, self) for vehicle_init_data in vehicle_init_data_list
        }

        # laneを初期化
        lane_init_data_list : list[dict[str, any]] = init_data["lane_init_data_list"]
        self.lane_dict : dict[int, Lane] = {
            lane_init_data["number"] : Lane(lane_init_data) for lane_init_data in lane_init_data_list
        }
        for lane in self.lane_dict.values() : 
            lane.update(self.vehicle_dict)

        # 設定
        self.limit_velocity = init_data["limit_velocity"]
        self.limit_accel = init_data["limit_accel"]
        self.limit_brake = init_data["limit_brake"]
        self.limit_step_count = init_data["limit_step_count"]

        # dqn
        self.dqn = dqn


    def start(self) -> None : 
        while self.judge_simulation_end() == False : 
            self.increment() 
        self.logger.write_log()


    def increment(self) -> None : 
        # 各vehicleが状況認識（内部の状態は変化しない）
        for vehicle in self.vehicle_dict.values() : 
            vehicle.recognize() 

        # 各vehicleが意思決定（更新はまだしない）
        for vehicle in self.vehicle_dict.values() : 
            vehicle.decide_action() 

        # ステップ数を更新
        self.step_count += 1

        # 各vehicleの状態を更新する
        for vehicle in self.vehicle_dict.values() : 
            vehicle.update() 

        # 各laneの状態を更新する
        for lane in self.lane_dict.values() : 
            lane.update(self.vehicle_dict)

        # 各vehicleが経験を格納
        for vehicle in self.vehicle_dict.values() : 
            vehicle.push_experience()

        # NNを更新
        self.dqn.optimize()


    def judge_simulation_end(self) -> bool : 
        # 時間がかかりすぎた場合強制終了
        over_limit_step_count = self.step_count >= self.limit_step_count

        # 全ての車がゴールしたら終了
        all_vehicle_goal = True
        for vehicle in self.vehicle_dict.values() : 
            all_vehicle_goal = all_vehicle_goal and vehicle.is_goal

        return over_limit_step_count or all_vehicle_goal 


    def get_lane_length(self, lane_index) -> int : 
        return self.lane_dict[lane_index].length
    
    
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
            front_vehicle_distance -= self.vehicle_dict[vehicle.number].length / 2 - \
                                        self.vehicle_dict[front_vehicle_number].length / 2
            return {
                "distance" : front_vehicle_distance, 
                "velocity" : self.vehicle_dict[front_vehicle_number].velocity, 
                "accel" : self.vehicle_dict[front_vehicle_number].accel
            }
        

    def calculate_reward(self, state_t0 : dict[str, any], state_t1 : dict[str, any]) -> float : 
        reward = 0 

        # 速度ボーナス
        # reward += state_t1["velocity"] ** 1.5
        
        # 速度制限
        # reward -= state_t1["over_velocity"] * ((state_t1["velocity"] - self.limit_velocity) ** 2)

        # 速度制御
        reward -= (state_t1["velocity"] - self.limit_velocity) ** 2

        # 加速度制限
        reward -= (max(0, state_t1["accel"] - self.limit_accel) ** 2) * 100
        reward -= (max(0, self.limit_brake - state_t1["accel"]) ** 2) * 100

        # 停止
        reward -= state_t1["is_stop"] * 10000

        # ターン毎の減点
        reward -= 1

        return reward

