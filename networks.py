from torch import nn
import torch

uniform = torch.distributions.Uniform(-100, 100)

class Linear(nn.Module):
    def __init__(self, batch_norm = False, dropout = False):
        super(Linear, self).__init__()
        self.IN = 128      # input vector length
        
        in_l  = self.IN       # input vector length
        out_l = int(in_l*1.25)           # ouput vector length; number of neurons
        LAYERS = 5
        layers = []

        for _ in range(LAYERS):
            layers.append(nn.Linear(in_l, out_l))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_l))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(0.5, inplace=False))
            in_l = out_l
            out_l = int(out_l*1.25)
        layers.append(nn.Linear(in_l, in_l))

        self.net = nn.Sequential(*layers)

    def gen_batch(self, B):
        return uniform.sample((B, self.IN))


    def forward(self, x):
        return self.net(x)
        

class Conv(nn.Module):
    def __init__(self, batch_norm = False, dropout = False):
        super(Conv, self).__init__()
        in_C  = 3     # input channels
        out_C = 16    # output channels
        LAYERS = 5
        layers = []

        for _ in range(LAYERS):
            layers.append(nn.Conv2d(in_C, out_C, kernel_size=3, stride=2, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_C))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout2d(0.5, inplace=False))
            in_C = out_C
            out_C *= 2
        layers.append(nn.Conv2d(in_C, out_C, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def gen_batch(self, B):
        return uniform.sample((B, 3, 128, 128))

    def forward(self, x):
        return self.net(x)


class RNN(nn.Module):
    def __init__(self, batch_norm = False, dropout = False):
        super(RNN, self).__init__()
        self.IN = 256
        
        in_f  = self.IN   # input features
        out_f = 128       # output features
        self.LAYERS = 5
        self.lstms, self.bns = [], []
        self.relus, self.drops = [], []

        for _ in range(self.LAYERS):
            self.lstms.append(nn.LSTM(in_f, out_f).cuda())
            if batch_norm:
                self.bns.append(nn.BatchNorm1d(out_f))
            self.relus.append(nn.ReLU())
            if dropout:
                self.drops.append(nn.Dropout(0.5, inplace=False))
            in_f = out_f
            out_f //= 2
        self.lstms.append(nn.LSTM(in_f, in_f).cuda())
        for i in range(len(self.lstms)):
            self.add_module(f'lstm{i}', self.lstms[i])
        for i in range(len(self.bns)):
            self.add_module(f'bn{i}', self.bns[i])
        for i in range(len(self.relus)):
            self.add_module(f'relu{i}', self.relus[i])
        for i in range(len(self.drops)):
            self.add_module(f'drop{i}', self.drops[i])

    def gen_batch(self):
        # batch dimension is second https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        return uniform.sample((16, B, self.IN))

    def forward(self, x):
        for i in range(self.LAYERS):
            x, _ = self.lstms[i](x)
            if self.bns:
                x = self.bns[i](x)
            x = self.relus[i](x)
            if self.drops:
                x = self.drops[i](x)
        x, _ = self.lstms[-1](x)
        return x




        
