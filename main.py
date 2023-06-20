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
        features = dataset[columns].astype('float32')
        labels = dataset[['Price']].astype('float32')

        # Scale values
        self.features = StandardScaler().fit_transform(features)
        self.labels = StandardScaler().fit_transform(labels)

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
        self.loss_func = nn.MSELoss()

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
        if (epoch+1) % 5 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))    

def evaluate(model, val_loader):
    validation_steps_outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(validation_steps_outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
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
    print(f'model: {model}')
    print(f'model num params: {sum(p.numel() for p in model.parameters())}')

    epochs = 30
    lr = 1e-2
    history1 = fit(epochs, lr, model, train_loader, test_loader)

    val_loss = [x['val_loss'] for x in history1]

    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.show()


    return 0

if __name__ == '__main__':
    sys.exit(main())
