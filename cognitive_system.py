from collections import deque

from extrinsic_module import ExtrinsicModule
from intrinsic_module import IntrinsicModule
from params import Params
from world_model import WorldModelNN, WorldModel, SteroidWorldModelNN


class CognitiveSystem:
    def __init__(self, device):
        memory = deque(maxlen=Params.Intrinsic.memory_size.value)
        input_world_memory = deque(maxlen=Params.World.mem_in_size.value)
        output_world_memory = deque(maxlen=Params.World.mem_out_size.value)
        extrinsic_memory = deque(maxlen=Params.Extrinsic.mem_size.value)

        self.intrinsic = IntrinsicModule(
            n=Params.Intrinsic.n.value,
            norm=Params.Intrinsic.norm.value,
            memory=memory,
        )
        self.extrinsic = ExtrinsicModule(
            nn_input_size=Params.Extrinsic.input_nn_size.value,
            nn_hidden_size=Params.Extrinsic.hidden_nn_size.value,
            nn_output_size=Params.Extrinsic.output_nn_size.value,
            lr=Params.Extrinsic.lr.value,
            num_epochs=Params.Extrinsic.epochs.value,
            batch_size=Params.Extrinsic.batch_size.value,
            memory=extrinsic_memory,
            device=device,
            in_path=Params.Extrinsic.nn_in_path.value,
            out_path=Params.Extrinsic.nn_out_path.value,
            debug=Params.Extrinsic.debug.value,
            train_set_size=Params.Extrinsic.train_set_size.value,
        )
        self.world = WorldModel(
            actions=Params.Action.actions.value,
            debug=Params.World.debug.value,
        )

        self.world_nn = SteroidWorldModelNN(
            actions=Params.Action.actions.value,
            nn_input_size=Params.World.input_nn_size.value,
            nn_hidden_size=Params.World.hidden_nn_size.value,
            nn_output_size=Params.World.output_nn_size.value,
            lr=Params.World.lr.value,
            num_epochs=Params.World.epochs.value,
            batch_size=Params.World.batch_size.value,
            memory_in=input_world_memory,
            memory_out=output_world_memory,
            device=device,
            in_path=Params.World.nn_in_path.value,
            out_path=Params.World.nn_out_path.value,
            debug=Params.World.debug.value,
            train_set_size=Params.World.train_set_size.value,
        )