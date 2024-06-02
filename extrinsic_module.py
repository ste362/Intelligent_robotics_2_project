import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from params import Colors


class ExtrinsicModule:

    def __init__(self,
                 nn_input_size=3,
                 nn_hidden_size=64,
                 nn_output_size=1,
                 memory=deque(maxlen=50), lr=0.001, num_epochs=10,
                 in_path=None, out_path=None, device='cpu', debug=False,
                 batch_size=2, train_set_size=100):
        self.nn = NeuralNetwork(nn_input_size, nn_hidden_size, nn_output_size).to(device)
        self.nn_output_size = nn_output_size
        self.optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.num_epochs = num_epochs
        self.memory_in = memory
        self.memory_out = deque(maxlen=memory.maxlen)
        self.mem_size = memory.maxlen
        self.out_path = out_path
        self.in_path = in_path
        self.debug = debug
        self.device = device
        self.batch_size = batch_size
        self.train_set_size = train_set_size
        self.load()

    def reset_memory(self):
        self.memory = deque(maxlen=self.mem_size)

    def train(self, X, y):
        if self.debug: print("\nStart Train Extrinsic Model ....")
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X, y)
        sampler = torch.utils.data.RandomSampler(dataset, num_samples=min(len(y), self.train_set_size))
        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

        # Addestrare la rete
        for epoch in range(self.num_epochs):
            for i, (inputs, target) in enumerate(train_loader):
                # Forward pass
                inputs = inputs.to(self.device)
                target = target.to(self.device)

                outputs = self.nn(inputs)
                loss = self.criterion(outputs, torch.reshape(target, (len(target), self.nn_output_size)))

                # Backward pass e ottimizzazione
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.debug and (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        if self.debug: print("Extrinsic train completed!\n")

    def load(self):
        if self.in_path is not None and os.path.exists(self.in_path):
            self.nn.load_state_dict(torch.load(self.in_path, map_location=torch.device(self.device)))
            self.nn.eval()
            print(f'{Colors.OKGREEN}Model {self.in_path} correctly loaded for EXTRINSIC module!{Colors.ENDC}')
        else:
            print(f'{Colors.FAIL}No model loaded for EXTRINSIC module!{Colors.ENDC}')

    def save(self, n=0):
        if self.out_path is not None:
            torch.save(self.nn.state_dict(), f'{self.out_path[:-3]}_{n}.pt')
        else:
            print(f'No save function enabled for Extrinsic Model!')

    def get_action(self, predicted_states):
        utility = []
        for predicted_state in predicted_states:
            input_state = predicted_state[3:6]
            input_state = torch.tensor(input_state,dtype=torch.float32)
            input_state = input_state.to(self.device)
            out = self.nn(input_state)
            out = out.cpu()
            utility.append(out.detach().numpy())
        #if self.debug: print("Neural output", utility)

        return np.argmax(utility)




class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.relu = nn.ReLU()  # activation function
        self.hidden = nn.Linear(input_size, hidden_size)  # hidden
        self.hidden2 = nn.Linear(hidden_size, hidden_size)  # hidden2
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

