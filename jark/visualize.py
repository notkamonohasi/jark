import matplotlib.pyplot as plt
import pandas as pd 
import os

def write_velocity_graph(vehicle_log_path : str, vehicle_number : int) -> None : 
    delta_t = 0.2
    vehicle_log = pd.read_csv(vehicle_log_path)
    velocity_list = vehicle_log["velocity"].to_list() 
    time_list = [t * delta_t for t in range(len(velocity_list))]
    plt.plot(time_list, velocity_list)
    plt.xlabel("time[s]", fontsize=14)
    plt.ylabel("velocity[m/s]", fontsize=14)
    plt.hlines(20, time_list[0], time_list[-1], colors="r", linewidth=3)
    plt.grid()
    os.system("mkdir -p " + vehicle_log_path + "/../graph")
    plt.savefig(vehicle_log_path + "/../graph/number_" + str(vehicle_number))
    plt.clf()

