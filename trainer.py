import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score

from tqdm import tqdm

# Define loss function and optimizer

def train(model, num_epochs, train_dataloader, loss_function=None, optimizer=None, device='cpu'):
    if loss_function == None:
        loss_function = nn.L1Loss()
        
    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=3)
    
    epoch_loss_logger = []
    print("\t Training progress:")
    # Training loop
    for epoch in range(num_epochs):
        # training
        train_loss = []
        model.train()
        
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
            loss = loss1 + loss2
            train_loss.append(loss)
            
            # zero the grad
            optimizer.zero_grad()
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        epoch_loss_logger.append(torch.mean(torch.tensor(train_loss)))
        
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
            pred_white_elo, pred_black_elo = model(input_sequence)[-1].squeeze()
            
            white_elo_predictions.append([target_white_elo.item(), pred_white_elo.item(), loss_function(target_white_elo, pred_white_elo).item()])
            black_elo_predictions.append([target_black_elo.item(), pred_black_elo.item(), loss_function(target_black_elo, pred_black_elo).item()])

    return white_elo_predictions, black_elo_predictions