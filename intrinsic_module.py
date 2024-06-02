from collections import deque

import numpy as np

from params import Colors


class IntrinsicModule:


    def __init__(self, memory=deque(maxlen=500), n=1, norm = 1):
        self.n = n
        self.norm = norm
        self.memory = memory

    def compute(self, s_k):
        novelty = 0
        pos_X = [x[0] for x in self.memory]
        pos_Y = [x[1] for x in self.memory]
        pos_Obj = [x[4] for x in self.memory]
        size = [x[5] for x in self.memory]
        theta = [x[2] for x in self.memory]
        ir = [x[3] for x in self.memory]

        max_x = max(pos_X); min_x = min(pos_X)
        max_y = max(pos_Y); min_y = min(pos_Y)
        max_pos = max(pos_Obj); min_pos = min(pos_Obj)
        max_size = max(size); min_size = min(size)
        max_theta = max(theta); min_theta = min(theta)
        max_ir = max(ir); min_ir = min(ir)


        for x in self.memory:

            contrib_theta = 0
            if abs(x[2] - s_k[2]) > 180:
                contrib_theta = 360 - abs(x[2] - s_k[2])
                contrib_theta = (contrib_theta - min_theta) / (2*(max_theta - min_theta) + 1)
            else:
                contrib_theta = ((abs(x[2] - s_k[2]) - min_theta) / (2 * (max_theta - min_theta) + 1))


            novelty += (
                (abs(x[0] - s_k[0]) - min_x)/(2*(max_x-min_x) + 1) ** self.norm +
                (abs(x[1] - s_k[1]) - min_y)/(2*(max_y-min_y) + 1) ** self.norm +
                contrib_theta ** self.norm +
                (abs(x[3] - s_k[3]) - min_ir)/(2*(max_ir-min_ir) + 1) ** self.norm +
                (abs(x[4] - s_k[4]) - min_pos)/(2*(max_pos-min_pos) + 1) ** self.norm +
                (abs(x[5] - s_k[5]) - min_size)/(2*(max_size-min_size) + 1) ** self.norm
            )/6 ** self.n


        return novelty / len(self.memory)

    def get_action(self, predicted_states):
        novelty_array = []
        #print("predicted states", predicted_states)

        for p in predicted_states:
            novelty_array.append(self.compute(p))

        num = np.argmax(novelty_array)
        print("novelty array [", end="")
        for i, e in enumerate(novelty_array):
            if i == num:
                print(f'{Colors.OKGREEN}{e:.5f}{Colors.ENDC}', end=", ")
            else:
                print(f'{e:.5f}', end=", ")
        print("]")
        return np.argmax(novelty_array)
