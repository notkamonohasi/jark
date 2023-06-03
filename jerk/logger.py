
import os
import pandas as pd
from statistics import mean
from pathlib import Path
import matplotlib.pyplot as plt

class EpisodeLogger : 
    def __init__(self, init_data : dict[str, any]) -> None:
        self.episode_path : Path = init_data["episode_path"]
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
        os.system("mkdir -p " + str(self.episode_path.joinpath("vehicle")))

    
    def write_log_vehicle(self, vehicle_number, vehicle_log_list : list[dict[str, any]]) : 
        # 空だとバグる
        if len(vehicle_log_list) == 0 : 
            return 
        
        columns = vehicle_log_list[0].keys()
        data_dict = {
            col : [vehicle_log_list[i][col] for i in range(len(vehicle_log_list))] for col in columns
        }
        df = pd.DataFrame(data_dict)
        df.to_csv(self.episode_path.joinpath("vehicle" + "/number_" + str(vehicle_number) + ".csv"))


class TotalLogger : 
    def __init__(self, sim_path : Path) -> None:
        self.sim_path : Path = sim_path

        self.reward_record : dict[int, list[float]] = {}

    def register_reward(self, episode, reward) -> None : 
        if episode not in self.reward_record : 
            self.reward_record[episode] = []
        self.reward_record[episode].append(reward)

    def write_result(self) -> None : 
        # 得られた報酬の推移を描画
        episode_list = self.reward_record.keys()
        episode_list = sorted(episode_list)
        meaned_reward_record = [mean(self.reward_record[i]) for i in episode_list]
        normalized_reward_record = self.get_normalize_list(meaned_reward_record)
        time_list = [i for i in range(len(normalized_reward_record))]
        plt.plot(time_list, normalized_reward_record)
        plt.xlabel("episode", fontsize = 14)
        plt.ylabel("reward", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.grid()
        plt.savefig(self.sim_path.joinpath("reward.png"), bbox_inches="tight")
        plt.clf()

    # 移動平均を計算
    def get_normalize_list(self, target_list : list[float], size = 100) -> list[float] : 
        ret = []
        pos_sum = sum(target_list[: size])
        for i in range(len(target_list) - size) : 
            pos_sum = pos_sum - target_list[i] + target_list[i + size]
            ret.append(pos_sum / size)
        return ret

