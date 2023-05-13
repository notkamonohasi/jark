from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .vehicle import Vehicle

from typing import Union
from util import exit_failure


class Lane : 
    def __init__(self, init_data : dict[str, Union[int, float]]) -> None:
        self.number = init_data["number"]
        self.length = init_data["length"]
        self.on_vehicle_list = []


    def update(self, vehicle_dict : dict[str, Vehicle]) : 
        # このレーンの上にいる車を登録する
        self.on_vehicle_list = [vehicle for vehicle in vehicle_dict.values() if self.number == vehicle.lane_number]
        
        # 前から順番に並び替える
        self.on_vehicle_list = sorted(self.on_vehicle_list, key=lambda vehicle: vehicle.lane_place, reverse=True)


    # lane上でvehicle_numberの前にいるvehicleの番号を取得する
    def get_front_vehicle_number(self, vehicle_number : int) -> int : 
        my_vehicle_index = None
        # 対象が先頭の時はNoneを返す
        for i, vehicle in enumerate(self.on_vehicle_list) :
            if vehicle.number == vehicle_number : 
                my_vehicle_index = i 
        
        if my_vehicle_index == None : 
            exit_failure("not found vehicle_nubmer in Lane::get_fron_vehicle_number")
        
        # 自分が先頭の時はNoneを返す
        if my_vehicle_index == 0 : 
            return None
        else : 
            return self.on_vehicle_list[i - 1].number
        

    # レーンの一番後ろにいる車の番号を取得する
    def get_back_vehicle_number(self) -> int :
        if len(self.on_vehicle_list) == 0 : 
            return None 
        else : 
            return self.on_vehicle_list[-1].number  
           
                

