from lstm import *
from evaluator import *
from trainer import *
from preprocessor import *
from data import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt

import time
import datetime

print("Initializing model...")
# Initialize parameters
input_size = 1
hidden_size = 50
output_size = 2
num_epochs = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Initialize model
model = LSTM(input_size, hidden_size, output_size)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1)

# Create the train and test data
print("Creating train data...")
start = time.time()
dataset = create_dataset(2)

train_data = dataset[:1]
test_data = dataset[-1:]

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
print("Time for data creation: " + str(datetime.timedelta(seconds=(time.time() - start))))

# Train LSTM
print("Train LSTM")
epoch_losses, model = train(model, num_epochs, train_dataloader, device=device)

white_elo_predictions, black_elo_predictions = evaluate(model, test_dataloader, device=device)
for white_elo_prediction in white_elo_predictions:
    print(white_elo_prediction)
    
# for black_elo_prediction in black_elo_predictions:
#     print(black_elo_prediction)
def visualize_data(data, title):
  x_axis = np.arange(num_epochs) + 1
  plt.plot(x_axis, data)
  plt.title(title)
  plt.show()

visualize_data(epoch_losses, "Epoch losses")