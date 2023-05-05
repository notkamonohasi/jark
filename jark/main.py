
import os

from simulator import Simulator 
from DQN.DQN import DQN

if __name__ == "__main__" : 
    init_data = {
        "delta_t" : 0.2, 
        "result_path" : "./result", 
        "state_columns" : ["accel", "velocity", "over_velocity", "over_accel", "over_brake", "is_stop"], 
        "learning_rate" : 0.001, 
        "target_learning_rate" : 0.001, 
        "buffer_size" : 10000, 
        "jark_cand" : [-2, 0, 2],
        "batch_size" : 64,
        "gamma" : 0.995, 
        "max_episode" : 20000, 
        "limit_velocity" : 20, 
        "limit_accel" : 3, 
        "limit_brake" : -5, 
        "limit_step_count" : 300
    }

    # lane
    lane_init_data_list = []
    for lane_number in range(1) : 
        lane_init_data = {
            "number" : lane_number, 
            "length" : 200, 
        }
        lane_init_data_list.append(lane_init_data)
    init_data["lane_init_data_list"] = lane_init_data_list

    # vehicle
    vehicle_init_data_list = []
    for vehicle_number in range(1) : 
        vehicle_init_data = {
            "number" : vehicle_number, 
            "length" : 4.4, 
            "velocity" : 10, 
            "accel" : 0, 
            "jark" : 0, 
            "lane_number" : 0, 
            "lane_place" : 0, 
            "route_list" : [0], 
            "jark_cand" : init_data["jark_cand"], 
            "limit_velocity" : init_data["limit_velocity"], 
            "limit_accel" : init_data["limit_accel"], 
            "limit_brake" : init_data["limit_brake"]
        }
        vehicle_init_data_list.append(vehicle_init_data)
    init_data["vehicle_init_data_list"] = vehicle_init_data_list

    # dqnを初期化
    dqn = DQN({
        "state_columns" : init_data["state_columns"], 
        "buffer_size" : init_data["buffer_size"], 
        "learning_rate" : init_data["learning_rate"],
        "target_learning_rate" : init_data["target_learning_rate"],
        "jark_cand" : init_data["jark_cand"], 
        "batch_size" : init_data["batch_size"], 
        "gamma" : init_data["gamma"], 
        "max_episode" : init_data["max_episode"]
    })

    os.system("rm -rf ./result")

    for pos_episode in range(1, init_data["max_episode"] + 1) : 
        print(pos_episode)
        init_data["result_path"] = "./result/episode_" + str(pos_episode).zfill(4)
        dqn.pos_episode = pos_episode
        simulator = Simulator(init_data, dqn)
        simulator.start()