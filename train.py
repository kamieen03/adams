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
        self.net = self.net.cuda()
        self.experiment_name = str(net_type).split('.')[1].split('\'')[0] + " "
        self.experiment_name += "BatchNorm"*batch_norm + "Dropout"* dropout + f"Weight decay {weight_decay}"*(weight_decay != 0)

        # cache net's parameters for later reinitialization
        self.cache = net_type(batch_norm, dropout)
        self.cache.load_state_dict(self.net.state_dict())
        self.cache = self.cache.cuda()
        
        # load saved target network (to make sure all experiments for given architecture are done with the same target)
        self.target = net_type(False, False)
        self.target.load_state_dict(torch.load(target_path))
        self.target.eval()
        self.target = self.target.cuda()
        for param in self.target.parameters():
            param.requires_grad = False

        # create optimizers and loss
        self.optims = {
            'SGD': torch.optim.SGD(self.net.parameters(), 1e-3, momentum=0.9, weight_decay = weight_decay),
            'Adam': torch.optim.Adam(self.net.parameters(), 1e-3, weight_decay = weight_decay),
            'AdamW': torch.optim.AdamW(self.net.parameters(), 1e-3, weight_decay = weight_decay)
        }
        
        self.L2 = torch.nn.MSELoss().cuda()    # L2 since we are doing multi-variable regression

        # training constants
        self.EPOCHS = 2            # number of epochs
        self.TRAIN_SIZE = 100      # number of batches in epoch
        self.TEST_SIZE = 100        # number of batches in epoch
        self.B = 16                 # batch size
        self.train_set = torch.cat([self.net.gen_batch(self.B).unsqueeze(0) for _ in range(self.TRAIN_SIZE)]).cuda()
        self.test_set  = torch.cat([self.net.gen_batch(self.B).unsqueeze(0) for _ in range(self.TEST_SIZE)]).cuda()

    def train_all(self):
        train_data, test_data = {}, {}
        final = {}

        for name, optimizer in self.optims.items():
            self.net.load_state_dict(self.cache.state_dict())
            self.net.cuda()
            means_train, means_test = self._train(optimizer, name)
            train_data[name] = means_train
            test_data[name] = means_test
            final[name] = {}
            final[name]['train'] = means_train[-1]
            final[name]['test'] = means_test[-1]

        self.plot(train_data, test_data)
        with open(f'results/{self.experiment_name}.txt', 'w+') as f:
            f.write(str(final))

    def _train(self, optimizer, name):
        means_train, means_test = [], []
        self._init_test(means_train, means_test)
        for i in range(self.EPOCHS):
            losses = []
            print(f'Train Epoch {i} optimizer {name} ...')
            for j in range(self.TRAIN_SIZE):
                batch = self.train_set[j]
                yt = self.target(batch)
                optimizer.zero_grad()
                y = self.net(batch)
                loss = self.L2(y, yt)
                loss.backward()
                optimizer.step()
                losses.append(float(loss))
            means_train.append(np.array(losses).mean())

            losses = []
            print(f'Test Epoch {i} optimizer {name} ...')
            for j in range(self.TEST_SIZE):
                with torch.no_grad():
                    batch = self.test_set[j]
                    yt = self.target(batch)
                    y = self.net(batch)
                    loss = self.L2(y, yt)
                    losses.append(float(loss))
            means_test.append(np.array(losses).mean())

        return means_train, means_test

    def _init_test(self, means_train, means_test):
        losses = []
        print("Init tests ...")
        with torch.no_grad():
            for j in range(self.TRAIN_SIZE):
                batch = self.train_set[j]
                yt = self.target(batch)
                y = self.net(batch)
                loss = self.L2(y, yt)
                losses.append(float(loss))
            means_train.append(np.array(losses).mean())
            losses = []
            for j in range(self.TEST_SIZE):
                batch = self.test_set[j]
                yt = self.target(batch)
                y = self.net(batch)
                loss = self.L2(y, yt)
                losses.append(float(loss))
        means_test.append(np.array(losses).mean())


    def plot(self, train_dict, test_dict):
        plt.figure()
        colors = ['r', 'g', 'b']
        for color, (name, ys) in zip(colors, train_dict.items()):
            plt.plot(range(self.EPOCHS+1), ys, color, label=name)
        for color, (name, ys) in zip(colors, test_dict.items()):
            plt.plot(range(self.EPOCHS+1), ys, color+'--', label=name)
        plt.legend()
        plt.xlim(0, self.EPOCHS)
        plt.xticks(range(self.EPOCHS+1))
        plt.title(self.experiment_name)
        #plt.show()
        plt.savefig(f'images/{self.experiment_name}.jpg')


def main():
    Trainer(Conv, 'targets/conv.pth', batch_norm=False, dropout=False, weight_decay=0).train_all()
    Trainer(Linear, 'targets/linear.pth', batch_norm=True,  dropout=False, weight_decay=0).train_all()
    #t.train_all()
    #t = Trainer(Linear, 'targets/linear.pth', batch_norm=False, dropout=True,  weight_decay=0)
    #t.train_all()
    #t = Trainer(Linear, 'targets/linear.pth', batch_norm=False, dropout=False, weight_decay=0.01)
    #t.train_all()
    #t = Trainer(Linear, 'targets/linear.pth', batch_norm=False, dropout=False, weight_decay=0.001)
    #t.train_all()
    #t = Trainer(Linear, 'targets/linear.pth', batch_norm=False, dropout=False, weight_decay=0.0001)
    #t.train_all()

if __name__ == '__main__':
    main()

        
