
import torch
from torch import nn


class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, out_dim=1):
        super(LSTMDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_size=input_dim+1,
                            hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, condition.unsqueeze(1).repeat((1, x.shape[1], 1))], dim=2)
        h = self.lstm(z)[0][:, -1:]
        x = self.linear(h)
        return x
