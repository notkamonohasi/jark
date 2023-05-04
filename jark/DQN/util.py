from collections import namedtuple
import torch

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")