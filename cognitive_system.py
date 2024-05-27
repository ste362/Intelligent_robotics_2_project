from collections import deque

from extrinsic_module import ExtrinsicModule
from intrinsic_module import IntrinsicModule
from params import Params
from world_model import WorldModelNN, WorldModel


class CognitiveSystem:
    def __init__(self, device):
        memory = deque(maxlen=500)
        input_world_memory = deque(maxlen=500)
        extrinsic_memory = deque(maxlen=50)

        self.intrinsic = IntrinsicModule(
            n=1/2,
            norm=IntrinsicModule.Norm.L1,
            memory=memory,
        )
        self.extrinsic = ExtrinsicModule(
            lr=0.001,
            num_epochs=10,
            memory=extrinsic_memory,
            device=device,
            in_path='models/extrinsic/extrinsic_nn.pt',
            out_path='models/extrinsic/extrinsic_nn.pt',
        )
        self.world = WorldModel(
            lr=0.001,
            num_epochs=10,
            memory_in=input_world_memory,
            memory_out=memory,
            device=device,
            in_path='models/world/world_nn.pt',
            out_path='models/world/world_nn.pt',
        )
