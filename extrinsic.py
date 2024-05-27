import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class ExtrinsicModule:

    def __init__(self):
        self.nn = NeuralNetwork(4, 64, 1).to(device)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        self.num_epochs = 10
        self.weights_init_normal(self.nn)

    ## takes in a module and applies the specified weight initialization
    def weights_init_normal(self, m):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''
        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            y = m.in_features
            # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0, 1 / np.sqrt(y))
            # m.bias.data should be 0
            m.bias.data.fill_(0)


    def train(self, X, y):
        dataset = TensorDataset(X, y)
        #print(dataset)
        batch_size = 2
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Addestrare la rete
        for epoch in range(self.num_epochs):
            for i, (inputs, target) in enumerate(train_loader):
                # Forward pass
                inputs = inputs.to(device)
                target = target.to(device)

                outputs = self.nn(inputs)
                loss = self.criterion(outputs, torch.reshape(target, (len(target),1)))

                # Backward pass e ottimizzazione
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print("Addestramento completato!")

    def load(self, path):
        self.nn.load_state_dict(torch.load(path))
        self.nn.eval()


def generate_data():
    X = np.random.randn(100, 4).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    return torch.tensor(X), torch.tensor(y)


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


if __name__ == "__main__":
    X, y = generate_data()
    module = ExtrinsicModule()
    module.train(X, y)