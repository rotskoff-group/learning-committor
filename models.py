import torch
import torch.nn as nn


class CommittorNet(nn.Module):
    def __init__(self, d, n, unit=torch.relu, thresh=None, init_mode="meanfield"):
        super(CommittorNet, self).__init__()
        self.n = n
        self.d = d
        self.lin1 = nn.Linear(d, n, bias=True)
        self.unit = unit
        self.lin2 = nn.Linear(n, 1, bias=False)
        self.thresh = thresh
        self.initialize(mode=init_mode)

    def initialize(self, mode="meanfield"):
        if mode == "meanfield":
            self.lin2.weight.data = torch.randn(
                self.lin2.weight.data.shape) / self.n
            self.renormalize()

    def forward(self, x):
        x = x.view(-1, self.d)
        x = self.lin1(x)
        x = self.unit(x)
        if self.thresh is not None:
            return self.thresh(self.lin2(x))
        else:
            return self.lin2(x)

    def renormalize(self):
        self.lin1.weight.data /= torch.norm(self.lin1.weight.data,
                                            dim=1).reshape(self.n, 1)
