import torch.nn as nn
import torch.nn.functional as F

class BaseNN(nn.Module):

    def __init__(self, 
                input_dim=3,
                n_hidden=64,
                n_output=3,
                n_layer=3,):
        super(BaseNN, self).__init__()
        self.include_past = False
        self.input_dim = input_dim
        self.n_hidden = n_hidden

        # Input Layer
        self.input = nn.Linear(input_dim, n_hidden)

        # Constructing the hidden layers with a modulelist
        self.fcs = []
        for i in range(n_layer):
            self.fcs.append(nn.Linear(n_hidden, n_hidden))
        self.fcs = nn.ModuleList(self.fcs)

        # Prediction Layer
        self.predict = nn.Linear(n_hidden, n_output)

class PSNN(BaseNN):
    """
    Params
    ----------
    n_visible:  int, number of past timesteps to pass into the model (including the current timestep).
                n = 1 reduces this model to SimplePredictor
    """
    def __init__(self, n_visible=3, n_output=3, n_layer=3, input_dim=3):
        super(PSNN, self).__init__(n_output=n_output, n_layer=n_layer, input_dim=input_dim)
        self.n_visible = n_visible
        # Here we alter the dimension in the input layer
        self.input = nn.Linear(self.input_dim * (self.n_visible+1), self.n_hidden)

    def forward(self, X):
        res = F.relu(self.input(X))
        for fc in self.fcs:
            res = F.relu(fc(res))
        res = self.predict(res)
        return res

class SimplePredictor(BaseNN):
    def __init__(self, input_dim, n_hidden, n_output, n_layer, activation=None):
        super(SimplePredictor, self).__init__(
            input_dim=input_dim,
            n_hidden=n_hidden,
            n_output=n_output,
            n_layer=n_layer,
        )
        self.activation = activation
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, X):
        res = F.relu(self.input(X))
        if self.activation is None:
            for fc in self.fcs:
                res = fc(res)
        else:
            for fc in self.fcs:
                res = F.relu(self.dropout(fc(res)))
        res = self.predict(res)
        return res