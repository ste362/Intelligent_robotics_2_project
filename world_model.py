import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from params import Colors


class WorldModelNN:

    def __init__(self,
                 actions,
                 nn_input_size=8,
                 nn_hidden_size=64,
                 nn_output_size=6,
                 train_set_size=500,
                 lr=0.001, num_epochs=10, memory_in=deque(maxlen=500),
                 memory_out=deque(maxlen=500), in_path=None,
                 out_path=None, device='cpu', debug=False,
                 batch_size=2):
        self.nn = NeuralNetwork(nn_input_size, nn_hidden_size, nn_output_size).to(device)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.criterion = nn.L1Loss()
        self.num_epochs = num_epochs
        self.actions = actions
        self.memory_in = memory_in
        self.memory_out = memory_out
        self.in_path = in_path
        self.out_path = out_path
        self.debug = debug
        self.device = device
        self.batch_size = batch_size
        self.train_set_size = train_set_size
        self.nn_output_size = nn_output_size
        self.load()

    def train(self, X, y):

        print("\n\nStart Train World Model ....")

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
                loss = self.criterion(outputs, torch.reshape(target, (len(target),  self.nn_output_size)))

                # Backward pass e ottimizzazione
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 1 == 0:
                print(
                    f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print("Addestramento completato!\n\n")

    def load(self):
        if self.in_path is not None and os.path.exists(self.in_path):
            self.nn.load_state_dict(torch.load(self.in_path, map_location=torch.device(self.device)))
            self.nn.eval()
            print(f'{Colors.OKGREEN}Model {self.in_path} correctly loaded for WORLD module!{Colors.ENDC}')
        else:
            print(f'{Colors.FAIL}No model loaded for WORLD module!{Colors.ENDC}')

    def save(self, n=0):
        if self.out_path is not None:
            torch.save(self.nn.state_dict(), f'{self.out_path[:-3]}_{n}.pt')
        else:
            print(f'No save function enabled for World Model!')

    def predict(self, state):
        input_states = []
        predicted_states = []
        #delta=[]
        for action in range(len(self.actions)):
            if action == 2 and state[3] > 20:
                print(f"{Colors.UNDERLINE}Muro{Colors.ENDC}")
            else:
                ## create one hot vector for the action
                one_hot = [0 for _ in range(len(self.actions))]
                one_hot[action] = 1

                input_state = state[:]
                input_state.extend(one_hot)
                input_states.append(input_state)

                input_state = torch.tensor(input_state, dtype=torch.float32)

                input_state = input_state.to(self.device)
                out = self.nn(input_state)
                out = out.cpu().detach().numpy()
                #delta.append(out)
                #out = [a + b for a, b in zip(state, out)]
                predicted_states.append(out)

        return input_states, predicted_states#, delta


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.relu = nn.ReLU()  # activation function
        self.hidden = nn.Linear(input_size, hidden_size+128)  # hidden
        self.hidden2 = nn.Linear(hidden_size+128, hidden_size)  # hidden2
        self.hidden3 = nn.Linear(hidden_size, hidden_size)  # hidden2
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.output(x)
        return x






# Mathematic World Model
class WorldModel:
    def __init__(self,
                 actions,
                 debug=False, **kwarg):
        self.debug = debug
        self.actions = actions

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
                new_x = x + 60 * np.cos(angle)
                new_y = y + 60 * np.sin(angle)
                if red_size > 0:
                    predicted_states.append([new_x, new_y, angle, ir, red_pos, red_size + 50])
                elif state[3] < 15:  # muro
                    predicted_states.append([new_x, new_y, angle, ir, red_pos, red_size])

            else:
                alpha = self.actions[action]
                new_theta = theta + alpha

                if new_theta > 359:
                    new_theta -= 360
                elif new_theta < 0:
                    new_theta += 360


                predicted_states.append([x, y, new_theta, ir, red_pos, red_size])

        return input_states, predicted_states





















































class SteroidWorldModelNN:

    def __init__(self,
                 actions,
                 nn_input_size=8,
                 nn_hidden_size=64,
                 nn_output_size=6,
                 train_set_size=500,
                 lr=0.001, num_epochs=10, memory_in=deque(maxlen=500),
                 memory_out=deque(maxlen=500), in_path=None,
                 out_path=None, device='cpu', debug=False,
                 batch_size=2):
        self.nn_pos = SteroidNeuralNetwork(nn_input_size, nn_hidden_size, nn_output_size).to(device)
        self.nn_sens = SteroidNeuralNetwork(nn_input_size, nn_hidden_size, nn_output_size).to(device)


        self.optimizer_pos = optim.Adam(self.nn_pos.parameters(), lr=lr)
        self.optimizer_sens = optim.Adam(self.nn_sens.parameters(), lr=lr)

        self.criterion_pos = nn.L1Loss()
        self.criterion_sens = nn.L1Loss()

        self.num_epochs = num_epochs
        self.actions = actions
        self.memory_in = memory_in
        self.memory_out = memory_out
        self.in_path = in_path
        self.out_path = out_path
        self.debug = debug
        self.device = device
        self.batch_size = batch_size
        self.train_set_size = train_set_size
        self.nn_output_size = nn_output_size
        self.load()

    def train(self, X, y):

        print("\n\nStart Train World Model ....")
        X = X.copy()
        y = y.copy()

        X_pos = [x[0:3]+x[6:11] for x in X]
        y_pos = [e[0:3] for e in y]
        self.train_pos(X_pos, y_pos)
        print("----------------------------------")
        X_sens = [x[3:6]+x[6:11] for x in X]
        y_sens = [e[3:6] for e in y]
        self.train_pos(X_sens, y_sens)

        print("Addestramento completato!\n\n")

    def train_pos(self, X, y):
        self._train(X, y, self.criterion_pos, self.optimizer_pos, self.nn_pos)

    def train_sens(self, X, y):
        self._train(X, y, self.criterion_sens, self.optimizer_sens, self.nn_sens)

    def _train(self, X, y, criterion, optimizer, nn):
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

                out = nn(inputs)
                loss = criterion(out, torch.reshape(target, (len(target), 3)))

                # Backward pass e ottimizzazione
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 1 == 0:
                print(
                    f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.2f}')

    def load(self):
        if os.path.exists(f'{self.in_path[:-3]}_pos.pt') and os.path.exists(f'{self.in_path[:-3]}_sens.pt'):
            self.nn_pos.load_state_dict(torch.load(f'{self.in_path[:-3]}_pos.pt', map_location=torch.device(self.device)))
            self.nn_pos.eval()

            self.nn_sens.load_state_dict(
                torch.load(f'{self.in_path[:-3]}_sens.pt', map_location=torch.device(self.device)))
            self.nn_sens.eval()
            print(f'{Colors.OKGREEN}Model {self.in_path[:-3]}_pos.pt correctly loaded for WORLD module!{Colors.ENDC}')
            print(f'{Colors.OKGREEN}Model {self.in_path[:-3]}_sens.pt correctly loaded for WORLD module!{Colors.ENDC}')
        else:
            print(f'{Colors.FAIL}No model loaded for WORLD module! {Colors.ENDC}')

    def save(self, n=0):
        if self.out_path is not None:
            torch.save(self.nn_pos.state_dict(), f'{self.out_path[:-3]}_{n}_pos.pt')
            torch.save(self.nn_sens.state_dict(), f'{self.out_path[:-3]}_{n}_sens.pt')
        else:
            print(f'No save function enabled for World Model!')
        pass

    def predict(self, state):
        input_states = []
        predicted_states = []
        for action in range(len(self.actions)):
            if action == 2 and state[3] > 15:
                print(f"{Colors.UNDERLINE}Muro{Colors.ENDC}")
            else:
                ## create one hot vector for the action
                one_hot = [0 for _ in range(len(self.actions))]
                one_hot[action] = 1




                input_pos = state[0:3]
                input_sens = state[3:6]

                input_pos.extend(one_hot)
                input_sens.extend(one_hot)


                input_pos = torch.tensor(input_pos, dtype=torch.float32).to(self.device)
                input_sens = torch.tensor(input_sens, dtype=torch.float32).to(self.device)


                out_pos = self.nn_pos(input_pos).cpu()
                out_sens = self.nn_sens(input_sens).cpu()

                predicted_state = np.concatenate((out_pos.detach().numpy(), out_sens.detach().numpy()), axis=0)

                predicted_states.append(predicted_state)

                input_state = state[:]
                input_state.extend(one_hot)
                input_states.append(input_state)

        return input_states, predicted_states


class SteroidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.relu = nn.ReLU()  # activation function

        self.input = nn.Linear(8, 64)  # hidden
        self.hidden1 = nn.Linear(64, 64)  # hidden
        self.output = nn.Linear(64, 3)  # hidden

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.output(x)
        return x














