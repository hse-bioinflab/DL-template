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
                    nn.Dropout(0.5),
                    nn.Linear(in_features=400, out_features=100, bias=True),
                    nn.Sigmoid(),
                    nn.Dropout(0.5),
                    nn.Linear(in_features=100, out_features=2, bias=True)
        )
    
    def forward(self, x):
        x, (h_n, c_n) = self.rnn(x)
        x = self.seq(x)
        return F.log_softmax(x, dim=-1)
