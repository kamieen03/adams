#!/usr/bin/env python3
import torch
import numpy as np
from networks import Linear, Conv, RNN
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, net_type, target_path, batch_norm, dropout, weight_decay):
        # create new network for training
        self.net = net_type(batch_norm, dropout)
        self.net.train()
        self.net_name = str(net_type).split('.')[1].split('\'')[0]

        # cache net's parameters for later reinitialization
        self.cache = net_type(batch_norm, dropout)
        self.cache.load_state_dict(self.net.state_dict())
        
        # load saved target network (to make sure all experiments for given architecture are done with the same target)
        self.target = net_type(False, False)
        self.target.load_state_dict(torch.load(target_path))
        self.target.eval()
        for param in self.target.parameters():
            param.requires_grad = False

        # create optimizers and loss
        self.optims = {
            'SGD': torch.optim.SGD(self.net.parameters(), 1e-3, momentum=0.9, weight_decay = weight_decay),
            'Adam': torch.optim.Adam(self.net.parameters(), 1e-3, weight_decay = weight_decay),
            'AdamW': torch.optim.AdamW(self.net.parameters(), 1e-3, weight_decay = weight_decay)
        }
        
        self.L2 = torch.nn.MSELoss()    # L2 since we are doing multi-variable regression

        # training constants
        self.EPOCHS = 10          # number of epochs
        self.TRAIN_SIZE = 100     # number of batches in epoch
        self.TEST_SIZE = 10       # number of batches in epoch
        self.B = 16               # batch size
        self.train_set = [self.net.gen_batch(self.B) for _ in range(self.TRAIN_SIZE)]
        self.test_set  = [self.net.gen_batch(self.B) for _ in range(self.TEST_SIZE)]

    def train_all(self):
        train_data, test_data = {}, {}
        for name, optimizer in self.optims.items():
            self.net.load_state_dict(self.cache.state_dict())
            means_train, means_test = self._train(optimizer, name)
            train_data[name] = means_train
            test_data[name] = means_test
        self.plot(train_data, test_data)
        

    def _train(self, optimizer, name):
        means_train, means_test = [], []
        self._init_test(means_train, means_test)
        for i in range(self.EPOCHS):
            losses = []
            for j in range(self.TRAIN_SIZE):
                batch = self.train_set[j]
                yt = self.target(batch)
                optimizer.zero_grad()
                y = self.net(batch)
                loss = self.L2(y, yt)
                loss.backward()
                optimizer.step()

                losses.append(float(loss))
                print(f'Train Epoch {i} batch {j} optimizer {name}: {loss}')
            means_train.append(np.array(losses).mean())
            losses = []
            for j in range(self.TEST_SIZE):
                with torch.no_grad():
                    batch = self.test_set[j]
                    yt = self.target(batch)
                    y = self.net(batch)
                    loss = self.L2(y, yt)
                    losses.append(float(loss))
                    print(f'Test Epoch {i} batch {j} optimizer {name}: {loss}')
            means_test.append(np.array(losses).mean())
        return means_train, means_test

    def _init_test(self, means_train, means_test):
        losses = []
        for j in range(self.TRAIN_SIZE):
            with torch.no_grad():
                batch = self.train_set[j]
                yt = self.target(batch)
                y = self.net(batch)
                loss = self.L2(y, yt)
                losses.append(float(loss))
        means_train.append(np.array(losses).mean())
        losses = []
        for j in range(self.TEST_SIZE):
            with torch.no_grad():
                batch = self.test_set[j]
                yt = self.target(batch)
                y = self.net(batch)
                loss = self.L2(y, yt)
                losses.append(float(loss))
        means_test.append(np.array(losses).mean())


    def plot(self, train_dict, test_dict):
        colors = ['r', 'g', 'b']
        for color, (name, ys) in zip(colors, train_dict.items()):
            plt.plot(range(self.EPOCHS+1), ys, color, label=name)
        for color, (name, ys) in zip(colors, test_dict.items()):
            plt.plot(range(self.EPOCHS+1), ys, color+'--', label=name)
        plt.legend()
        plt.show()
        plt.savefig('temp.jpg')


def main():
    t = Trainer(Linear, 'targets/linear.pth', batch_norm=False, dropout=False, weight_decay=0)
    t.train_all()

if __name__ == '__main__':
    main()

        
