#!/usr/bin/env python3

import torch
from networks import Linear, Conv, RNN
from torchsummary import summary


def main():
    net = Linear().cuda()
    summary(net, (128,))
    torch.save(net.state_dict(), 'targets/linear.pth')
    net = Conv().cuda()
    summary(net, (3,256,256))
    torch.save(net.state_dict(), 'targets/conv.pth')
    net = RNN().cuda()
    print(net.state_dict())
    print(net(torch.randn(16,32,256).cuda()).shape)
    torch.save(net.state_dict(), 'targets/rnn.pth')

if __name__ == '__main__':
    main()
