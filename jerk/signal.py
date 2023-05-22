from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .simulator import Simulator

from typing import Union
from enum import Enum

class Aspect(Enum) : 
    BLUE = 0 
    YELLOW_TO_RED = 1
    RED = 2
    YELLOW_TO_BLUE = 3
    ASPECT_SIZE = 4

def convert_index_into_aspect(index : int) -> Aspect : 
    assert index <= 3
    if index == 0 : 
        return Aspect.BLUE
    elif index == 1 : 
        return Aspect.YELLOW_TO_RED
    elif index == 2 : 
        return Aspect.RED
    elif index == 3 : 
        return Aspect.YELLOW_TO_BLUE


class Signal : 
    def __init__(self, init_data : dict[str, any], simulator : Simulator) -> None:
        self.signal_number = init_data["signal_number"]
        self.first_time = init_data["first_time"]   # 時刻0での位相[s]
        self.interval_list : list[int] = init_data["interval_list"]   # 青 -> 黄 -> 赤 -> 黄
        
        self.simulator = simulator

        self.cycle = int(sum(self.interval_list))

        assert len(self.interval_list) == Aspect.ASPECT_SIZE

        self.update()   # 初期化しておく必要がある
    

    def update(self) : 
        # 剰余の計算が入るため、秒からミリ秒に変えて計算する
        pos_time = self.simulator.get_second()   # s
        amari = int((self.first_time + pos_time) * 1000) % int(self.cycle * 1000) 
        pos_sum = 0
        for index in range(len(self.interval_list)) : 
            pos_sum += self.interval_list[index] * 1000
            if pos_sum > amari : 
                self.signal_aspect : Aspect = convert_index_into_aspect(index)
                self.remain_second = (amari - pos_sum) / 1000   # msに戻す
                self.remain_second = round(self.remain_second, 3)   # 表示を綺麗にするため
                return 
            

    def get_signal_state(self) -> dict[str, Union(Aspect, int)] : 
        return {
            "signal_aspect" : self.signal_aspect, 
            "remain_second" : self.remain_second
        }

        




