from lstm import *
from evaluator import *
from trainer import *
from preprocessor import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import time
import datetime
import os

print("Initializing model...")
reuse_model = False
# Initialize model parameters
input_size = 1
hidden_size = 100
output_size = 2

# Initialize model
model = LSTM(input_size, hidden_size, output_size)
if reuse_model == True:
  model.load_state_dict(torch.load("model"))

# Initialize rest of the parameters and functions
data_size = 500
depth = 15
num_epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3)

# Check if evaluated_positions.json is empty, if yes then initialize
if os.stat("evaluated_positions.json").st_size == 0:
  initialize_evaluations("evaluated_positions.json")

# Create the train and test data
print("Creating train data...")
start = time.time()
dataset = create_dataset('filtered_output.pgn', data_size, depth=depth)

train_data, test_data = train_test_split(dataset, test_size=0.10)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
print("Time for data creation: " + str(datetime.timedelta(seconds=(time.time() - start))))

# Train LSTM
print("Train LSTM")
start = time.time()
epoch_losses, model = train(model, num_epochs, train_dataloader, device=device)
print("Time for training LSTM: " + str(datetime.timedelta(seconds=(time.time() - start))))

white_elo_predictions, black_elo_predictions = evaluate(model, test_dataloader, device=device)
for white_elo_prediction in white_elo_predictions:
  print(white_elo_prediction)
  
# save model
torch.save(model.state_dict(), "model")
    
# for black_elo_prediction in black_elo_predictions:
#     print(black_elo_prediction)

def visualize_data(data, title):
  x_axis = np.arange(num_epochs) + 1
  plt.plot(x_axis, data)
  plt.title(title)
  plt.show()

visualize_data(epoch_losses, "Epoch losses")

# TODO
# preprocess lichess games
# train on lichess games
# https://stackoverflow.com/questions/48124604/regex-to-extract-between-start-and-end-strings-and-match-the-entire-line-contain