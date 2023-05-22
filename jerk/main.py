
import os, random

from simulator import Simulator 
from DQN.DQN import DQN

if __name__ == "__main__" : 
    init_data = {
        "delta_t" : 0.2, 
        "result_path" : "./result", 
        "state_columns" : ["accel", "velocity", "distance_intersection"], 
        "learning_rate" : 0.0001, 
        "target_learning_rate" : 0.005, 
        "buffer_size" : 10000, 
        "jerk_cand" : [-2, 0, 2],
        "batch_size" : 128,
        "gamma" : 0.995, 
        "max_episode" : 5000, 
        "log_interval" : 100, 
        "limit_velocity" : 20, 
        "limit_accel" : 2, 
        "limit_brake" : -3, 
        "limit_step_count" : 300
    }

    # signal
    signal_init_data_list = [
        {
            "number" : 0, 
            "first_time" : 0, 
            "interval_list" : [5, 1, 5, 1]
        }
    ]
    init_data["signal_init_data_list"] = signal_init_data_list

    # intersection
    intersection_init_data_list = [
        {
            "number" : 0, 
            "y" : 0, 
            "x" : 0, 
            "signal_number" : None
        }, 
        {
            "number" : 1, 
            "y" : 0, 
            "x" : 400, 
            "signal_number" : 0
        }
    ]
    init_data["intersection_init_data_list"] = intersection_init_data_list

    # lane
    lane_init_data_list = []
    for lane_number in range(1) : 
        lane_init_data = {
            "number" : lane_number, 
            "from_intersection_number" : 0, 
            "to_intersection_number" : 1
        }
        lane_init_data_list.append(lane_init_data)
    init_data["lane_init_data_list"] = lane_init_data_list

    # vehicle
    vehicle_init_data_list = []
    for vehicle_number in range(1) : 
        vehicle_init_data = {
            "number" : vehicle_number, 
            "length" : 4.4, 
            "decide_action_way" : "DQN", 
            "velocity" : 2, 
            "accel" : 0, 
            "jerk" : 0, 
            "lane_number" : 0, 
            "lane_place" : 10 * 2 * vehicle_number,   # 適当
            "route_list" : [0], 
            "jerk_cand" : init_data["jerk_cand"], 
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
        "jerk_cand" : init_data["jerk_cand"], 
        "batch_size" : init_data["batch_size"], 
        "gamma" : init_data["gamma"], 
        "max_episode" : init_data["max_episode"]
    })

    os.system("rm -rf ./result")

    for pos_episode in range(1, init_data["max_episode"] + 1) : 
        print(pos_episode)
        init_data["pos_episode"] = pos_episode
        init_data["result_path"] = "./result/episode_" + str(pos_episode).zfill(4)
        dqn.pos_episode = pos_episode
        simulator = Simulator(init_data, dqn)
        simulator.start()