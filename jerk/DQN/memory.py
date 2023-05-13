
from collections import deque
import random

from .util import Transition 

class Memory(object):
    def __init__(self, init_data : dict[str, any]):
        self.memory = deque([], maxlen=init_data["buffer_size"])

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def get_all(self) : 
        return self.memory[:]

    def __len__(self):
        return len(self.memory)