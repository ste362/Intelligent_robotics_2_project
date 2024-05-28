from collections import deque

from extrinsic_module import ExtrinsicModule
from intrinsic_module import IntrinsicModule
from params import Params
from world_model import WorldModelNN, WorldModel


class CognitiveSystem:
    def __init__(self, device):
        memory = deque(maxlen=Params.World.mem_out_size.value)
        input_world_memory = deque(maxlen=Params.World.mem_in_size.value)
        extrinsic_memory = deque(maxlen=Params.Extrinsic.mem_size.value)

        self.intrinsic = IntrinsicModule(
            n=Params.Intrinsic.n.value,
            norm=Params.Intrinsic.norm.value,
            memory=memory,
        )
        self.extrinsic = ExtrinsicModule(
            lr=Params.Extrinsic.lr.value,
            num_epochs=Params.Extrinsic.epochs.value,
            batch_size=Params.Extrinsic.batch_size.value,
            memory=extrinsic_memory,
            device=device,
            in_path=Params.Extrinsic.nn_in_path.value,
            out_path=Params.Extrinsic.nn_out_path.value,
            debug=Params.Extrinsic.debug.value,
        )
        self.world = WorldModel(
            lr=Params.World.lr.value,
            num_epochs=Params.World.epochs.value,
            batch_size=Params.World.batch_size.value,
            memory_in=input_world_memory,
            memory_out=memory,
            device=device,
            in_path=Params.World.nn_path.value,
            out_path=Params.World.nn_path.value,
            debug=Params.World.debug.value,
        )

        self.world_nn = WorldModelNN(
            lr=Params.World.lr.value,
            num_epochs=Params.World.epochs.value,
            batch_size=Params.World.batch_size.value,
            memory_in=input_world_memory,
            memory_out=memory,
            device=device,
            in_path=Params.World.nn_path.value,
            out_path=Params.World.nn_path.value,
            debug=Params.World.debug.value,
        )