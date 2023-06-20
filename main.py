# Inspired by https://medium.com/@avinashshah099/playing-with-linear-regression-using-pytorch-af339da6a4b9

import sys
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cpu")

columns = [
    'Avg. Area Income',
    'Avg. Area House Age',
    'Avg. Area Number of Rooms',
    'Avg. Area Number of Bedrooms',
    'Area Population',
] 

class Dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file):
        # Read the data
        dataset = pd.read_csv(csv_file)

        # Pick featuers and labels
        features = dataset[columns]
        labels = dataset[['Price']]

        # Scale values
        features = StandardScaler().fit_transform(features)
        labels = StandardScaler().fit_transform(labels)

        # Move the data to the GPU
        self.features = torch.from_numpy(features).float().to(device)
        self.labels = torch.from_numpy(labels).float().to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(len(columns), 1),
        )
        # self.loss_func = nn.MSELoss()

        self.loss_func = nn.L1Loss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch):
        X, y_true = batch
        return F.l1_loss(self(X), y_true)

    def validation_step(self, batch):
        X, y_true = batch
        y_pred = self(X)
        loss = self.loss_func(y_pred, y_true)
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, validation_step_outputs):
        batch_losses = [x['val_loss'] for x in validation_step_outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        epoch += 1
        if epoch % 5 == 0 or epoch == num_epochs-1:
            print(f'Epoch [{epoch:3}], train_loss: {result["train_loss"]:.4f} val_loss: {result["val_loss"]:.4f}')

def evaluate(model, val_loader):
    validation_steps_outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(validation_steps_outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        batch_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            batch_losses.append(loss.detach())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(batch_losses).mean()
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history

def main() -> int:

    # Prep the data
    batch_size = 64
    dataset = Dataset("./USA_housing.csv")
    train_data, test_data = train_test_split(dataset, test_size=0.2)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = Model()
    model.to(device)
    print(f'model: {model}')
    print(f'model num params: {sum(p.numel() for p in model.parameters())}')

    epochs = 100
    lr = 1e-3
    history1 = fit(epochs, lr, model, train_loader, test_loader)

    train_loss = [x['train_loss'] for x in history1]
    val_loss = [x['val_loss'] for x in history1]

    plt.plot(val_loss, label='val_loss')
    plt.plot(train_loss, label='train_loss')
    plt.legend()
    plt.show()


    return 0

if __name__ == '__main__':
    sys.exit(main())
