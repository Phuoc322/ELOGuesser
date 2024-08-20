import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

# Define loss function and optimizer

def train(model, num_epochs, loss_function=None, optimizer=None, train_dataloader=None, device='CPU'):
    if loss_function == None:
        loss_function = nn.MSELoss()
        
    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=0.01)        
    
    # Example target tensor (next score prediction)
    # In a real scenario, you'd have the actual next scores
    # target_tensor = input_tensor[1:]  # Shifted by one
    # input_tensor = input_tensor[:-1]  # Same length as target

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1} of {num_epochs}")
        # training
        train_loss = []
        model.train()
        print("\t Training progress: \n")
        
        #TODO change embedding length with a sequence of moves and ELO
        for embedding, length, label in tqdm(train_dataloader):
            optimizer.zero_grad()
            train_prediction = model(embedding[0])
            loss = loss_function(train_prediction[-1].squeeze(dim=0), torch.tensor(float(label[0]), device=device))
            train_loss.append(loss)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        epoch_loss_logger.append(torch.mean(torch.tensor(train_loss)))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
    print("\n\t Evaluation progress: \n")
    
def evaluate(model, num_epochs, loss_function=None, optimizer=None, test_dataloader=None, device='CPU'):
    with torch.no_grad():
        model.eval()
        predictions = []
        targets = []

        #TODO change embedding length with a sequence of moves and ELO
        for embedding, length, label in tqdm(test_dataloader):
            test_prediction = round(model(embedding[0])[-1].item())
            predictions.append(float(test_prediction))
            targets.append(int(label[0]))


        train_f1_score = f1_score(predictions, targets)
        print("\t Test F1 Score in epoch " + str(epoch) + ": " + str(train_f1_score)
            + " Train loss: " + str(epoch_loss_logger[epoch].item()))
        epoch_f1_scores.append(train_f1_score)

    return epoch_loss_logger, epoch_f1_scores