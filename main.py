from lstm import *
from evaluator import *
from trainer import *

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

# Initialize the model
input_size = 1
hidden_size = 50
num_layers = 1
output_size = 1

# Initialize model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create the train and test data
print("Create train data")
train_evals = evaluate_positions_from_pgn_string("1. g4 e5 2. f3 Qh4#")
train_dataloader = DataLoader(train_evals, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
print("Create test data")
#testloader = DataLoader(test_data_set, batch_size=5, shuffle=False, num_workers=0, drop_last=True)

train(model=model, num_epochs=1, train_dataloader=train_dataloader)