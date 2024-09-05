from lstm import *
from evaluator import *
from trainer import *
from preprocessor import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import time
import datetime
import os

print("Initializing model...")
reuse_model = True
test_only = True
# Initialize model parameters
input_size = 1
hidden_size = 100

# Initialize model
model = LSTM(input_size, hidden_size)
if reuse_model:
  model.load_state_dict(torch.load("model"))

# Initialize rest of the parameters and functions
data_size = 1030
depth = 15
num_epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2)

# Check if evaluated_positions.json is empty, if yes then initialize
if os.stat("evaluated_positions.json").st_size == 0:
  initialize_evaluations("evaluated_positions.json")

# only create train data, if train and test is desired, tests in any case
if not test_only:
  # Create the train and test data
  print("Creating train data...")
  start = time.time()
  dataset = create_dataset('data\output.pgn', data_size, depth=depth)
  train_data, test_data = train_test_split(dataset, test_size=0.10)
  train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
  
  test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
  print("Time for data creation: " + str(datetime.timedelta(seconds=(time.time() - start))))

# Train LSTM
  print("Train LSTM")
  start = time.time()
  if not reuse_model:
    epoch_losses, model = train(model, num_epochs, train_dataloader, device=device)
    #visualize_data(epoch_losses, epochs, "Epoch losses")
  print("Time for training LSTM: " + str(datetime.timedelta(seconds=(time.time() - start))))

  white_elo_predictions, black_elo_predictions = evaluate(model, test_dataloader, device=device)
  for white_elo_prediction in white_elo_predictions:
    print(white_elo_prediction)
  
else:
  print("Test LSTM")
  # iterate through files in test games
  for f in os.listdir("test_games"):
    filename = os.fsdecode(f)
    # evaluate each game
    test_data = create_dataset('test_games\\' + filename, data_size, depth=depth)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    pred_white_elo, pred_black_elo = evaluate(model, test_dataloader, device=device)
    print("White ELO: " + str(pred_white_elo))
    print("Black ELO: " + str(pred_black_elo))

# save model
torch.save(model.state_dict(), "model")

# TODO
# data analysis, check the elo of white and black with histogram
# look over model architecture, LSTM and NN, and training process, stochastic minibatch