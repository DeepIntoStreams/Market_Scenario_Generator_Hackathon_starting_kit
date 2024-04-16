"""
This is a sample file, user must provide a python function named init_generator() which:
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


class GeneratorBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorBase, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward_(self, batch_size: int, ts_length: int, device: str):
        """ Implement here generation scheme. """
        # ...
        pass

    def forward(self, batch_size: int, n_lags: int, device: str):
        x = self.forward_(batch_size, n_lags, device)
        x = self.pipeline.inverse_transform(x)
        return x


class TSGenerator(GeneratorBase):
    """
    Sample generator class
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_layers: int, init_fixed: bool = True):
        super(TSGenerator, self).__init__(input_dim, output_dim)
        # LSTM
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self.linear.apply(init_weights)

        self.init_fixed = init_fixed

    def forward(self, batch_size: int, ts_length: int, device: str) -> torch.Tensor:
        z = (0.1 * torch.randn(batch_size, ts_length,
                               self.input_dim)).to(device)  # cumsum(1)

        if self.init_fixed:
            h0 = torch.zeros(self.rnn.num_layers, batch_size,
                             self.rnn.hidden_size).to(device)
        else:
            h0 = torch.randn(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(
                device).requires_grad_()
        z[:, 0, :] *= 0
        z = z.cumsum(1)
        c0 = torch.zeros_like(h0)
        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(h1)

        assert x.shape[1] == ts_length
        return x


def init_generator():
    print("Initialisation of the model.")
    config = {
        "G_hidden_dim": 64,
        "G_input_dim": 5,
        "G_num_layers": 2
    }
    generator = TSGenerator(input_dim=config["G_input_dim"],
                            hidden_dim=config["G_hidden_dim"],
                            output_dim=4,
                            n_layers=config["G_num_layers"],
                            init_fixed=True)
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
        fake_data = generator(batch_size=2000, ts_length=20, device='cpu')
    print(fake_data[0, 0:10, :])
