from torch import nn
import torch
import torch.nn.functional as F


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.0):
        super(MixerBlock, self).__init__()
        self.dropout = dropout

        self.layerNorm1 = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(num_patch, token_dim)
        self.dropout1 = nn.Dropout(self.dropout)
        self.lin2 = nn.Linear(token_dim, num_patch)
        self.dropout2 = nn.Dropout(self.dropout)

        self.layerNorm2 = nn.LayerNorm(dim)
        self.lin3 = nn.Linear(dim, channel_dim)
        self.dropout3 = nn.Dropout(self.dropout)
        self.lin4 = nn.Linear(channel_dim, dim)
        self.dropout4 = nn.Dropout(self.dropout)


    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x
    
    def token_mix(self, x):
        x = self.layerNorm1(x)
        x = torch.transpose(x, -1, -2)
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.lin2(x)
        x = self.dropout2(x)
        return torch.transpose(x, -1, -2)
    
    def channel_mix(self, x):
        x = self.layerNorm2(x)
        x = self.lin4(self.dropout3(F.gelu(self.lin3(x))))
        return self.dropout4(x)