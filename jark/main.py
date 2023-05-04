from simulator import Simulator 

if __name__ == "__main__" : 
    init_data = {}

    init_data["delta_t"] = 0.2 
    init_data["result_path"] = "./result"

    # lane
    lane_init_data_arr = []
    for lane_number in range(1) : 
        lane_init_data = {
            "number" : lane_number, 
            "length" : 100, 
        }
        lane_init_data_arr.append(lane_init_data)
    init_data["lane_init_data_arr"] = lane_init_data_arr

    # vehicle
    vehicle_init_data_arr = []
    for vehicle_number in range(1) : 
        vehicle_init_data = {
            "number" : vehicle_number, 
            "length" : 4.4, 
            "velocity" : 15, 
            "accel" : 0, 
            "jark" : 0, 
            "lane_number" : 0, 
            "lane_place" : 0, 
            "route_arr" : [0], 
        }
        vehicle_init_data_arr.append(vehicle_init_data)
    init_data["vehicle_init_data_arr"] = vehicle_init_data_arr

    simulator = Simulator(init_data)
    simulator.start()