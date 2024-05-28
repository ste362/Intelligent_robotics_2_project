from aenum import Enum, skip


class Params(Enum):
    error_margin_rotation = 4.5  # degrees
    eps = 0.4
    eps_decr = 0.0
    n_iterations_before_stop = 700
    use_neural_world_model = True

    @skip
    class State(Enum):
        posx = 0
        posy = 1
        theta = 2
        ir_val = 3
        obj_posx = 4
        obj_size = 5

    @skip
    class Action(Enum):
        actions = [-30, -15, 0, 15, 30]
        left_m = 0  # -30 degree
        left_s = 1  # -15 degree
        forward = 2  #
        right_s = 3  # +15 degree
        right_m = 4  # +30 degree

    @skip
    class World(Enum):
        lr = 0.001
        epochs = 10
        batch_size = 4

        input_nn_size = 8
        output_nn_size = 3
        hidden_nn_size = 64

        mem_in_size = 500
        mem_out_size = 500

        train = True
        nn_in_path = 'models/world/world_nn_0.pt'
        nn_out_path = 'models/world/world_nn.pt'

        debug = False

    @skip
    class Extrinsic(Enum):
        lr = 0.001
        epochs = 10
        batch_size = 2

        input_nn_size = 3
        output_nn_size = 1
        hidden_nn_size = 64

        mem_size = 50

        train = False
        nn_in_path = 'models/extrinsic/extrinsic_nn_with_math_world_working.pt'
        nn_out_path = 'models/extrinsic/eliminami.pt'

        debug = False

    @skip
    class Intrinsic(Enum):
        n = 2
        norm = 2

        debug = False

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
