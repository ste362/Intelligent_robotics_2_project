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
        self.criterion = nn.MSELoss()
        self.num_epochs = 10

    def train(self, X, y):
        dataset = TensorDataset(X, y)
        #print(dataset)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Addestrare la rete
        for epoch in range(self.num_epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                # Forward pass
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.nn(inputs)
                loss = self.criterion(outputs, labels)

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
    y = np.random.randn(100, 5).astype(np.float32)

    return torch.tensor(X), torch.tensor(y)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.relu = nn.ReLU()  # activation function
        self.hidden = nn.Linear(input_size, hidden_size)  # hidden
        #self.hidden2 = nn.Linear(hidden_size, hidden_size)  # hidden2
        self.output = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)

        #x = self.hidden2(x)
        #x = self.relu(x)

        x = self.output(x)
        return x


if __name__ == "__main__":
    X, y = generate_data()
    module = ExtrinsicModule()
    module.train(X, y)