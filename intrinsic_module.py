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
            x_diff, y_diff, _ = self.subtract(x, s_k)
            if x_diff < 10 and y_diff < 10:  # se si è mosso di poco vuol dire che si è girato
                novelty += (0.05 * np.abs(x[2] - s_k[2])) ** self.n
                if x[5] > 0 and np.abs(x[2] - s_k[2]) <= 20:
                    novelty += x[5] * 1/len(self.memory) * 20/(np.abs(x[2] - s_k[2])+1)
            else:
                #novelty += (abs(x[0] - s_k[0]) ** self.norm.value + abs( x[1] - s_k[1]) ** self.norm.value + abs(x[5]-s_k[5])) ** self.n  # + 0.0008*np.abs(x[2]-new_state[2])
                novelty += (abs(x[0] - s_k[0]) ** self.norm.value + abs(x[1] - s_k[1]) ** self.norm.value) ** self.n

        if s_k[5] > 0:
            novelty += s_k[5]*((len(self.memory))**1/4)

        return novelty / len(self.memory) if len(self.memory) > 0 else 1

    def get_action(self, predicted_states):
        novelty_array = []
        #print("predicted states", predicted_states)
        for p in predicted_states:
            novelty_array.append(self.compute(p))

        print("novelty array", novelty_array, "memory_len:", len(self.memory))
        return np.argmax(novelty_array)
