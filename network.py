import torch
from torch import nn

class RNN(nn.Module):

    def __init__(self,INPUT_SIZE,HIDDEN_SIZE):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,     # rnn hidden unit
            num_layers=1,       # number of rnn layer

        )
        self.out = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        t, b, f = x.shape
        x = x.view ( t * b, f )
        x = self.out ( x )
        x = x.view ( t, b, -1 )
        return x
