
import os
import pandas as pd

class Logger : 
    def __init__(self, init_data : dict[str, any]) -> None:
        self.result_path = init_data["result_path"]
        self.pos_episode = init_data["pos_episode"]
        self.log_interval = init_data["log_interval"]

        self.vehicle_log_dict : dict[int, list[dict[str, any]]] = {}


    def register_vehicle_log(self, vehicle_number, vehicle_log : dict[int, any]) : 
        if vehicle_number not in self.vehicle_log_dict : 
            self.vehicle_log_dict[vehicle_number] = []
        self.vehicle_log_dict[vehicle_number].append(vehicle_log)


    def write_log(self) : 
        # 規定回数毎のみ
        if self.pos_episode % self.log_interval != 0 : 
            return

        # 結果用のディレクトリを作る
        self.make_result_dir()

        # vehicle
        for vehicle_number, vehicle_log_list in self.vehicle_log_dict.items() : 
            self.write_log_vehicle(vehicle_number, vehicle_log_list)

    
    def make_result_dir(self) : 
        os.system("mkdir -p " + self.result_path + "/vehicle")

    
    def write_log_vehicle(self, vehicle_number, vehicle_log_list : list[dict[str, any]]) : 
        # 空だとバグる
        if len(vehicle_log_list) == 0 : 
            return 
        
        columns = vehicle_log_list[0].keys()
        data_dict = {
            col : [vehicle_log_list[i][col] for i in range(len(vehicle_log_list))] for col in columns
        }
        df = pd.DataFrame(data_dict)
        df.to_csv(self.result_path + "/vehicle" + "/number_" + str(vehicle_number) + ".csv")


