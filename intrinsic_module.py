from collections import deque

import numpy as np
from enum import Enum


class IntrinsicModule:

    class Norm(Enum):
        L1 = 1
        L2 = 2

    def __init__(self, memory = deque(maxlen=500), n=1, norm:Norm=Norm.L1):
        self.n = n
        self.norm = norm
        self.memory = memory

    def subtract(self, t1, t2):
        return (round(abs(t1[0] - t2[0]), 1), round(abs(t1[1] - t2[1]), 1), round(abs(t1[2] - t2[2]), 1))

    def compute(self, s_k):
        novelty = 0
        for x in self.memory:
            x_diff, y_diff, _ = self.subtract(x, s_k)
            #print(x_diff, y_diff)
            if x_diff < 10 and y_diff < 10: # se si è mosso di poco vuol dire che si è girato
                novelty += (0.01 * np.abs(x[2] - s_k[2])) ** self.n
            else:
                #novelty += (abs(x[0] - s_k[0]) ** self.norm.value + abs( x[1] - s_k[1]) ** self.norm.value + abs(x[5]-s_k[5])) ** self.n  # + 0.0008*np.abs(x[2]-new_state[2])
                novelty += ((x[0] - s_k[0]) ** self.norm.value + (x[1] - s_k[1]) ** self.norm.value) ** self.n


        return novelty / len(self.memory) if len(self.memory)>0 else 1


    def get_action(self, predict_state):
        novelty_array = []
        for p in predict_state:
            novelty_array.append(self.compute(p))

        #print("novelty array",novelty_array)

        return np.argmax(novelty_array)