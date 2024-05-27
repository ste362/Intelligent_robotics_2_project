from enum import Enum
from intrinsic_module import IntrinsicModule


class Params(Enum):
    class State(Enum):
        posx = 0,
        posy = 1,
        ir_val = 3,
        obj_size = 5

    class Action(Enum):
        left_m = 0,  # -30 degree
        left_s = 1,  # -15 degree
        forward = 2,  #
        right_s = 3,  # +15 degree
        right_m = 4  # +30 degree

    class World(Enum):
        nn_path = '',
        lr = 0.001,
        epochs = 10,
        batch_size = 2,

        input_nn_size = 3,
        output_nn_size = 1,
        hidden_nn_size = 64,

        mem_in_size = 500,
        mem_out_size = 500,

        debug = False,

    class Extrinsic(Enum):
        nn_path = '',
        lr = 0.001,
        epochs = 10,
        batch_size = 2,

        input_nn_size = 11,
        output_nn_size = 6,
        hidden_nn_size = 64,

        mem_size = 50,

        debug = False,

    class Intrinsic(Enum):
        n = 1 / 2,
        norm: IntrinsicModule.Norm = IntrinsicModule.Norm.L1

        debug = False,

    eps = 0.9
    eps_decr = 0.01
    n_iterations_before_stop = 1000
