
import os
import pandas as pd

class Logger : 
    def __init__(self, result_path) -> None:
        self.result_path = result_path

        self.vehicle_log_dict : dict[int, list[dict[str, any]]] = {}


    def register_vehicle_log(self, vehicle_number, vehicle_log : dict[int, any]) : 
        if vehicle_number not in self.vehicle_log_dict : 
            self.vehicle_log_dict[vehicle_number] = []
        self.vehicle_log_dict[vehicle_number].append(vehicle_log)


    def write_log(self) : 
        # 結果用のディレクトリを作る
        self.make_result_dir()

        # vehicle
        for vehicle_number, vehicle_log_arr in self.vehicle_log_dict.items() : 
            self.write_log_vehicle(vehicle_number, vehicle_log_arr)

    
    def make_result_dir(self) : 
        os.system("rm -rf " + self.result_path) 
        os.mkdir(self.result_path)
        os.mkdir(self.result_path + "/vehicle")

    
    def write_log_vehicle(self, vehicle_number, vehicle_log_arr : list[dict[str, any]]) : 
        if len(vehicle_log_arr) == 0 : 
            return 
        
        columns = vehicle_log_arr[0].keys()
        data_dict = {
            col : [vehicle_log_arr[i][col] for i in range(len(vehicle_log_arr))] for col in columns
        }
        df = pd.DataFrame(data_dict)
        df.to_csv(self.result_path + "/vehicle" + "/number_" + str(vehicle_number) + ".csv")


