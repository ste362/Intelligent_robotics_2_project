import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from params import Params


class WorldModelNN:

    def __init__(self, lr=0.001, num_epochs=10, memory_in=deque(maxlen=500), memory_out=deque(maxlen=500), path=None, device='cpu', debug=False, batch_size=2):
        self.nn = NeuralNetwork(11, 64, 6).to(device)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.num_epochs = num_epochs
        self.actions = [-30,-15,0,15,30]
        self.memory_in = memory_in
        self.memory_out = memory_out
        self.path = path
        if path is not None:
            self.load(path)
        self.debug = debug
        self.device = device
        self.batch_size = batch_size

    def train(self, X, y):

        print("Start Train World Model ....\n")
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X, y)
        #print(dataset)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Addestrare la rete
        for epoch in range(self.num_epochs):
            for i, (inputs, target) in enumerate(train_loader):
                # Forward pass
                inputs = inputs.to(self.device)
                target = target.to(self.device)

                outputs = self.nn(inputs)
                loss = self.criterion(outputs, torch.reshape(target, (len(target),6)))

                # Backward pass e ottimizzazione
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print("Addestramento completato!")

    def load(self, path=None):
        if path is not None and os.path.exists(path):
            self.nn.load_state_dict(torch.load(path))
            self.nn.eval()
        else:
            if self.debug: print(f'File {path} not exist!')

    def save(self, path=None):
        if path is not None and os.path.exists(path):
            torch.save(self.nn.state_dict(), path)
        elif self.path is not None and os.path.exists(self.path):
            torch.save(self.nn.state_dict(), self.path)
        else:
            if self.debug: print(f'File {path} not exist!')

    def predict(self, state):
        input_states=[]
        predicted_states = []
        for action in range(len(self.actions)):
            if  action==2 and state[3]>15:
                print("Muro")
            else:
                ## create one hot vector for the action
                one_hot = [0 for _ in range(len(self.actions))]
                one_hot[action] = 1

                input_state = state[:]
                input_state.extend(one_hot)
                input_states.append(input_state)


                input_state = torch.tensor(input_state, dtype=torch.float32)
                input_state=input_state.to(self.device)
                out = self.nn(input_state)
                out=out.cpu()
                predicted_states.append(out.detach().numpy())
                #print(list(out.detach().numpy()))

        return input_states,predicted_states


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.relu = nn.ReLU()  # activation function
        self.hidden = nn.Linear(input_size, hidden_size)  # hidden
        self.hidden2 = nn.Linear(hidden_size, hidden_size)  # hidden2
        self.output = nn.Linear(hidden_size, output_size)



    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x










# Mathematic World Model
class WorldModel:
    def __init__(self, debug=False, **kwarg):
        self.debug = debug
        self.actions = [-30,-15,0,15,30]

    def train(self, X, y):
        pass

    def load(self, path=None):
        pass

    def save(self, path=None):
        pass

    def predict(self, state):
        input_states = []
        predicted_states = []
        x, y, theta, ir, red_pos, red_size = state
        for action in range(len(self.actions)):
            if action == 2:  ## dritto
                angle = np.deg2rad(theta)
                new_x = x + 60 * np.sin(angle)
                new_y = y + 60 * np.cos(angle)
                if red_size > 0:
                    predicted_states.append([new_x, new_y, theta, ir ,red_pos,red_size+150])
                elif state[3] < 15:  # muro
                    predicted_states.append([new_x, new_y, theta, ir ,red_pos,red_size])

            else:
                new_angle = theta + action
                if new_angle > 359:
                    new_angle -= 360
                elif new_angle < 0:
                    new_angle += 360

                #se sto girando a destra

                new_red_pos=red_pos
                """
                d = 500 - ir
                if d < 0: d = 0

                alpha = abs(self.actions[action])
                correction = np.sin(np.deg2rad(alpha)) * d

                if action < 2:
                    new_red_pos -= correction
                else:
                    new_red_pos += correction

                if new_red_pos < 0 or new_red_pos > 100:
                    new_red_pos = 0
                """

                """
                if red_pos>0:

                        print(new_red_pos)
                        new_red_pos = max(0,min(red_pos-self.actions[action],100))
                        if new_red_pos==100:
                            new_red_pos = 0
                        print(new_red_pos,action)
                """
                predicted_states.append([x, y, new_angle, ir, new_red_pos, red_size])


        return input_states, predicted_states