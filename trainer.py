import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score

from tqdm import tqdm

from early_stopper import *

# Define loss function and optimizer

def train(model, num_epochs, train_dataloader, loss_function=None, optimizer=None, device='cpu'):
    if loss_function == None:
        loss_function = nn.L1Loss()
        
    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=1)
        
    early_stopper = EarlyStopper(patience=10, min_delta=10)
    
    epoch_loss_logger = []
    actual_num_epochs = 0
    print("\t Training progress:")
    # Training loop
    for epoch in range(num_epochs):
        # training
        train_loss = []
        model.train()
        # zero the grad
        optimizer.zero_grad()
        loss = 0
        num_samples = 0
        for (target_white_elo, target_black_elo), evals in (train_dataloader):
            # targets to float and remove one dimension otherwise pytorch complains
            target_white_elo = target_white_elo.float().squeeze()
            target_black_elo = target_black_elo.float().squeeze()
            
            # input sequence with additional dimension (seq_len, 1), 1 is input size, one eval at the time
            input_sequence = torch.tensor(evals)[:,None].float()
            pred_white_elo, pred_black_elo = model(input_sequence)[-1].squeeze()
            
            # calculate losses and then sum up
            loss1 = loss_function(pred_white_elo, target_white_elo)
            loss2 = loss_function(pred_black_elo, target_black_elo)
            loss += loss1 + loss2
            train_loss.append(loss)
            num_samples += 1
            
        # loss is the average loss across every sample
        loss = torch.div(loss, num_samples)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        epoch_loss_logger.append(torch.mean(torch.tensor(train_loss)))
        
        if early_stopper.early_stop(loss):
            print("Stopped after %s epochs" % epoch)
            actual_num_epochs = epoch
            break
        
    return epoch_loss_logger, model
    
def evaluate(model, test_dataloader, loss_function=None, optimizer=None, device='cpu'):
    if loss_function == None:
        loss_function = nn.L1Loss()
        
    with torch.no_grad():
        model.eval()
        white_elo_predictions = []
        black_elo_predictions = []
        print("\n\t Evaluation progress: \n")
        
        for (target_white_elo, target_black_elo), evals in (test_dataloader):
            # targets to float otherwise pytorch complains
            target_white_elo = target_white_elo.float().squeeze()
            target_black_elo = target_black_elo.float().squeeze()
            
            # input sequence with additional dimension (seq_len, 1), 1 is input size, one eval at the time
            input_sequence = torch.tensor(evals)[:,None].float()
            pred_white_elo, pred_black_elo = model(input_sequence)[-1].round()
            pred_white_elo = pred_white_elo.squeeze()
            pred_black_elo = pred_black_elo.squeeze()
            
            white_elo_predictions.append([target_white_elo.item(), pred_white_elo.item(), loss_function(target_white_elo, pred_white_elo).item()])
            black_elo_predictions.append([target_black_elo.item(), pred_black_elo.item(), loss_function(target_black_elo, pred_black_elo).item()])

    return white_elo_predictions, black_elo_predictions

def evaluate_on_sample(model, test_dataloader, loss_function=None, optimizer=None, device='cpu'):     
    with torch.no_grad():
        model.eval()
        for evals in (test_dataloader):
            # input sequence with additional dimension (seq_len, 1), 1 is input size, one eval at the time
            input_sequence = evals.clone().detach()[:,None].float()
            pred_white_elo, pred_black_elo = model(input_sequence)[-1].round()
            pred_white_elo = pred_white_elo.squeeze().long().item()
            pred_black_elo = pred_black_elo.squeeze().long().item()

    return pred_white_elo, pred_black_elo