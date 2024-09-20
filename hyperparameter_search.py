import os
import tempfile

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint

def train_lstm(config):
    data_size = 500

    dataset = create_dataset(pgn_file, data_size, depth=depth)
    
    train_data, test_data = train_test_split(dataset, test_size=0.10)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    model = LSTM(input_size, hidden_size)

    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    for i in range(10):
        train_func(model, optimizer, train_loader)
        acc = test_func(model, test_loader)
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if (i + 1) % 5 == 0:
                # This saves the model to the trial directory
                torch.save(
                    model.state_dict(),
                    os.path.join(temp_checkpoint_dir, "model.pth")
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            # Send the current training result back to Tune
            train.report({"mean_accuracy": acc}, checkpoint=checkpoint)
            
search_space = {
    "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
    "momentum": tune.uniform(0.1, 0.9),
}

# Uncomment this to enable distributed execution
# `ray.init(address="auto")`

# Download the dataset first
datasets.MNIST("~/data", train=True, download=True)

tuner = tune.Tuner(
    train_mnist,
    param_space=search_space,
)
results = tuner.fit()