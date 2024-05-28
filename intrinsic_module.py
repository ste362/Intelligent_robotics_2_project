from collections import deque

import numpy as np
from enum import Enum


class IntrinsicModule:
    class Norm(Enum):
        L1 = 1
        L2 = 2

    def __init__(self, memory=deque(maxlen=500), n=1, norm: Norm = Norm.L1):
        self.n = n
        self.norm = norm
        self.memory = memory

    def subtract(self, t1, t2):
        return abs(t1[0] - t2[0]), abs(t1[1] - t2[1]), abs(t1[2] - t2[2])

    def compute(self, s_k):
        novelty = 0
        for x in self.memory:
            #x_diff, y_diff, _ = self.subtract(x, s_k)

            novelty += ((x[0] - s_k[0]) ** self.norm.value + (x[1] - s_k[1]) ** self.norm.value +  9*(x[2] - s_k[2]) ** self.norm.value) ** self.n


        return novelty / len(self.memory)

    def get_action(self, predicted_states):
        novelty_array = []
        #print("predicted states", predicted_states)
        for p in predicted_states:
            novelty_array.append(self.compute(p))

        #print("novelty array", novelty_array, "memory_len:", len(self.memory))
        return np.argmax(novelty_array)
