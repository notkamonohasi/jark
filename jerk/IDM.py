from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .simulator import Simulator
    from .vehicle import Vehicle

import math


T = 1.5   # safe time headway
a = 0.73  # 加速度最大値
b = 1.67  # 減速度最大値
d = 4     # acceleration exponent 
s0 = 2    # 距離最小値

def get_jerk_by_IDM(vehicle : Vehicle) -> float : 
    v0 = vehicle.limit_velocity   # 希望速度
    v = vehicle.state["velocity"]
    dv = vehicle.state["velocity"] - vehicle.state["front_vehicle_velocity"]
    s = vehicle.state["front_vehicle_distance"]
    prev_accel = vehicle.state["accel"]
    delta_t = vehicle.simulator.delta_t
    s_star = s0 + v * T + v * dv / (2 * math.sqrt(a * b))
    next_accel = a * (1 - (v / v0) ** d - (s_star / s) ** 2)
    jerk = (next_accel - prev_accel) / delta_t   # 得られた加速度をjerkに変換
    return jerk


