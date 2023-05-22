from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .simulator import Simulator

from typing import Union

class Intersection : 
    def __init__(self, init_data : dict[str, any], simulator : Simulator) -> None:
        self.intersection_number = init_data["number"]
        self.y = init_data["y"]
        self.x = init_data["x"]
        self.signal_number = init_data["signal_number"]

        self.simulator = simulator

    
    def get_place(self) -> tuple[int, int] : 
        return (self.y, self.x)
    

    def get_signal_state(self) -> Union(None, dict[str, any]) :
        if self.signal_number == None : 
            return None 
        else : 
            return self.simulator.signal_dict[self.signal_number].get_signal_state()


