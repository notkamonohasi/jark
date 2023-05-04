from simulator import Simulator 

if __name__ == "__main__" : 
    init_data = {
        "delta_t" : 0.2, 
        "result_path" : "./result", 
        "state_columns" : ["accel", "velocity"], 
        "learning_rate" : 0.01, 
        "target_learning_rate" : 0.01, 
        "buffer_size" : 100, 
        "jark_cand" : [0, 1, 2],
        "batch_size" : 10,
        "gamma" : 0.99
    }

    # lane
    lane_init_data_list = []
    for lane_number in range(1) : 
        lane_init_data = {
            "number" : lane_number, 
            "length" : 100, 
        }
        lane_init_data_list.append(lane_init_data)
    init_data["lane_init_data_list"] = lane_init_data_list

    # vehicle
    vehicle_init_data_list = []
    for vehicle_number in range(1) : 
        vehicle_init_data = {
            "number" : vehicle_number, 
            "length" : 4.4, 
            "velocity" : 15, 
            "accel" : 0, 
            "jark" : 0, 
            "lane_number" : 0, 
            "lane_place" : 0, 
            "route_list" : [0], 
            "jark_cand" : init_data["jark_cand"]
        }
        vehicle_init_data_list.append(vehicle_init_data)
    init_data["vehicle_init_data_list"] = vehicle_init_data_list

    simulator = Simulator(init_data)
    simulator.start()