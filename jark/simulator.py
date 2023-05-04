

from logger import Logger
from vehicle import Vehicle
from lane import Lane

class Simulator : 
    def __init__(self, init_data : dict[str, any]) : 
        self.delta_t = init_data["delta_t"]
        self.step_count = 0 

        # loggerを初期化
        self.logger = Logger(init_data["result_path"])

        # vehicleを初期化
        vehicle_init_data_arr : list[dict[str, any]] = init_data["vehicle_init_data_arr"]
        self.vehicle_dict : dict[int, Vehicle] = {
            vehicle_init_data["number"] : Vehicle(vehicle_init_data, self) for vehicle_init_data in vehicle_init_data_arr
        }

        # laneを初期化
        lane_init_data_arr : list[dict[str, any]] = init_data["lane_init_data_arr"]
        self.lane_dict : dict[int, Lane] = {
            lane_init_data["number"] : Lane(lane_init_data) for lane_init_data in lane_init_data_arr
        }


    def start(self) -> None : 
        while self.judge_simulation_end() == False : 
            self.increment() 
        self.logger.write_log()


    def increment(self) -> None : 
        # 各vehicleが意思決定（更新はまだしない）
        for vehicle in self.vehicle_dict.values() : 
            vehicle.decide_action() 

        # ステップ数を更新
        self.step_count += 1

        # 各vehicleを更新する
        for vehicle in self.vehicle_dict.values() : 
            vehicle.update() 


    def judge_simulation_end(self) -> bool : 
        # 全ての車がゴールしたら終了
        all_vehicle_goal = True
        for vehicle in self.vehicle_dict.values() : 
            all_vehicle_goal = all_vehicle_goal and vehicle.is_goal
        return all_vehicle_goal


    def get_lane_length(self, lane_index) -> int : 
        return self.lane_dict[lane_index].length

