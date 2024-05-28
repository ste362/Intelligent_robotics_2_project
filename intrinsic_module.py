from collections import deque

import numpy as np


class IntrinsicModule:


    def __init__(self, memory=deque(maxlen=500), n=1, norm = 1):
        self.n = n
        self.norm = norm
        self.memory = memory

    def compute(self, s_k):
        novelty = 0
        for x in self.memory:
            novelty += (
                        abs(x[0] + s_k[0]) ** self.norm +
                        abs(x[1] - s_k[1]) ** self.norm +
                        abs(x[2] - s_k[2]) ** self.norm
                       ) ** self.n


        return novelty / len(self.memory)

    def get_action(self, predicted_states):
        novelty_array = []
        #print("predicted states", predicted_states)

        for p in predicted_states:
            novelty_array.append(self.compute(p))

        #print("novelty array", novelty_array, "memory_len:", len(self.memory))
        return np.argmax(novelty_array)
