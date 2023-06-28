import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

class G4Detector(nn.Module):
    def __init__(self, onehot_len):
        super(G4Detector, self).__init__()

        self.conv1 = nn.Conv1d(onehot_len, 80, 2)
        self.conv2 = nn.Conv1d(onehot_len, 80, 3)
        self.conv3 = nn.Conv1d(onehot_len, 96, 6)

        self.linear_block = nn.Sequential(nn.Linear(256, 32),
                                          nn.ReLU(),
                                          nn.Linear(32, 1),
                                          nn.Sigmoid())

    def forward(self, x):
        output = x.transpose(1, 2)

        output1 = self.conv1(output)
        output2 = self.conv2(output)
        output3 = self.conv3(output)

        output1, _  = torch.max(output1, 2)
        output2, _  = torch.max(output2, 2)
        output3, _  = torch.max(output3, 2)

        output = torch.cat([output1, output2, output3], dim=1)

        output = self.linear_block(output)
        return output
