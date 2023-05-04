
from typing import Union

class Lane : 
    def __init__(self, init_data : dict[str, Union[int, float]]) -> None:
        self.number = init_data["number"]
        self.length = init_data["length"]
