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

# Check if evaluated_positions.json is empty, if yes then initialize
if os.stat("evaluated_positions.json").st_size == 0:
  initialize_evaluations("evaluated_positions.json")

# only create train data, if train and test is desired, tests in any case
if not test_only:
  # Initialize rest of the parameters and functions
  data_size = 500
  depth = 15
  num_epochs = 1000
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  loss_function = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=2)

  # Create the train and test data
  print("Creating train data...")
  pgn_file = 'data\output_all_filter.pgn'
  start = time.time()
  dataset = create_dataset(pgn_file, data_size, depth=depth)

  # balance the dataset by elo values
  # bins = 15
  # dataset, w_count, b_count = balance_dataset(dataset, bins, data_size, pgn_file, depth)
  # visualize_dataset(dataset, bins)

  train_data, test_data = train_test_split(dataset, test_size=0.10)
  train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
  test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

  # save dataloader to get fast access
  torch.save(train_dataloader, "./dataset/train.pth")
  torch.save(test_dataloader, "./dataset/test.pth")

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
    test_data = create_dataset('test_games\\' + filename)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    pred_white_elo, pred_black_elo = evaluate(model, test_dataloader)
    print("White ELO: " + str(pred_white_elo))
    print("Black ELO: " + str(pred_black_elo))

# save model
torch.save(model.state_dict(), "model")

# TODO sorted by priority, if any task is too difficult, weaken the requirements or just skip task
# wollen eine Funktion haben die uns für ein komplettes PGN die vier wichtigen tags zurückgibt
# hyperparameter search
# balancing
#   ist im pgn als [TimeControl "600+0"] notiert, Zahl links vom + ist Gesamtzeit in Sekunden pro Spieler, Zahl rechts davon der Inkrement, den ein Spieler bekommt, nachdem er seinen Zug macht (kann man ignorieren)
# 4. data analysis, check the elo of white and black with histogram, balance dataset
#   algorithmic approach:
#     1. start with 1000 games
#     2. build the histogram with class bins of size 50
#     3. balance the data by adding games such that the maximum difference of the smallest and largest class has maximum difference of 10%
# 5. look over model architecture, LSTM and NN, and training process, stochastic minibatch