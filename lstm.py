import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False)
        
        # find the next number to the power of 2
        p = 1
        if (hidden_size and not(hidden_size & (hidden_size - 1))):
            p = hidden_size
        else:
            while (p < hidden_size):
                p <<= 1
        
        # Define the output layer
        self.fc = nn.Sequential(nn.Linear(hidden_size, p),
                                nn.Linear(p, 2))
        
    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out
