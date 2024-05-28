from aenum import Enum, skip
from intrinsic_module import IntrinsicModule


class Params(Enum):
    eps = 0.9
    eps_decr = 0.02
    n_iterations_before_stop = 300


    @skip
    class State(Enum):
        posx = 0
        posy = 1
        ir_val = 3
        obj_size = 5

    @skip
    class Action(Enum):
        left_m = 0  # -30 degree
        left_s = 1  # -15 degree
        forward = 2  #
        right_s = 3  # +15 degree
        right_m = 4  # +30 degree

    @skip
    class World(Enum):
        nn_path = ''
        lr = 0.001
        epochs = 10
        batch_size = 2

        input_nn_size = 3
        output_nn_size = 1
        hidden_nn_size = 64

        mem_in_size = 500
        mem_out_size = 500

        nn_in_path = 'models/world/world_nn.pt'
        nn_out_path = 'models/world/world_nn.pt'

        debug = False

    @skip
    class Extrinsic(Enum):
        nn_path = ''
        lr = 0.001
        epochs = 10
        batch_size = 2

        input_nn_size = 11
        output_nn_size = 6
        hidden_nn_size = 64

        mem_size = 50

        nn_in_path = 'models/extrinsic/extrinsic_nn_9.pt'
        nn_out_path = 'models/extrinsic/extrinsic_nn.pt'

        debug = False

    @skip
    class Intrinsic(Enum):
        n = 1/4
        norm: IntrinsicModule.Norm = IntrinsicModule.Norm.L2

        debug = False
