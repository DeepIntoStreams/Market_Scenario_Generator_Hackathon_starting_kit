"""
This is a sample file. Any user must provide a python function named init_generator() which:
    - initializes an instance of the generator,
    - loads the model parameters from model_dict.py,
    - returns the model.
"""
import numpy as np
import os
import pickle
import torch
import torch.nn as nn

print(os.path.abspath(__file__))
PATH_TO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dict.pkl')
PATH_TO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake.pkl')


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(         m.weight.data, gain=nn.init.calculate_gain('relu'))
        try:
            nn.init.zeros_(m.bias)
        except:
            pass


# Sample model based on LSTM
class GeneratorBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorBase, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, batch_size: int, n_lags: int, device: str):
        """ Implement here generation scheme. """
        # ...
        pass


class ConditionalLSTMGenerator(GeneratorBase):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_layers: int):
        super(ConditionalLSTMGenerator, self).__init__(input_dim, output_dim)
        # LSTM
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self.linear.apply(init_weights)


    def forward(self, batch_size: int, condition: torch.Tensor, n_lags: int, device: str) -> torch.Tensor:
        z = (0.1 * torch.randn(batch_size, n_lags,
                               self.input_dim - condition.shape[-1])).to(device)  # cumsum(1)
        z[:, 0, :] *= 0  # first point is fixed
        z = z.cumsum(1)
        z = torch.cat([z, condition.unsqueeze(1).repeat((1, n_lags, 1))], dim=2)

        h0 = torch.zeros(self.rnn.num_layers, batch_size,
                         self.rnn.hidden_size).to(device)

        c0 = torch.zeros_like(h0)
        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(h1)

        assert x.shape[1] == n_lags
        return x


def init_generator():
    print("Initialisation of the model.")
    config = {
        "G_hidden_dim": 64,
        "G_input_dim": 5,
        "G_num_layers": 2
    }
    generator = ConditionalLSTMGenerator(input_dim=config["G_input_dim"],
                            hidden_dim=config["G_hidden_dim"],
                            output_dim=10,
                            n_layers=config["G_num_layers"])
    print("Loading the model.")
    # Load from .pkl
    with open(PATH_TO_MODEL, "rb") as f:
        model_param = pickle.load(f)
    generator.load_state_dict(model_param)
    generator.eval()
    return generator


if __name__ == '__main__':
    generator = init_generator()
    print("Generator loaded. Generate fake data.")
    with torch.no_grad():
        condition = torch.ones([200, 1])
        fake_data = generator(batch_size=200, condition=condition, n_lags=20, device='cpu')
    print(fake_data[0, 0:10, :])
