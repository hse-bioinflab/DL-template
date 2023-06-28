from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from IPython.display import clear_output

class DeepZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
                    nn.Conv1d(1062, 400, kernel_size=(3,), stride=(1,), padding=(1,)),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False),
                    nn.Conv1d(800, 600, kernel_size=(5,), stride=(1,), padding=(2,)),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False),
                    nn.LSTM(600, 400, bidirectional=True),
                    nn.Dropout(0.5),
                    nn.Linear(in_features=800, out_features=200, bias=True)
                    nn.Sigmoid(),
                    nn.Dropout(0.5),
                    nn.Linear(in_features=200, out_features=2, bias=True)
        )
    
    def forward(self, x):
        x, (h_n, c_n) = self.rnn(x)
        x = self.seq(x)
        return F.log_softmax(x, dim=-1)
