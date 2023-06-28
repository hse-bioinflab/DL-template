from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from IPython.display import clear_output

class DeepZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(1062, 500, num_layers=2, bidirectional=True)
        self.seq = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features=1000, out_features=500, bias=True),
                    nn.Sigmoid(),
                    nn.Dropout(0.5),
                    nn.Linear(in_features=500, out_features=2, bias=True)
        )
    
    def forward(self, x):
        x, (h_n, c_n) = self.rnn(x)
        x = self.seq(x)
        return F.log_softmax(x, dim=-1)
