import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import os
from tqdm import tqdm
from visdom import Visdom
import numpy as np

from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn

from dataset import SiameseWhaleDataset
from model import SiameseNetwork
from config import Configure
from utils import PairwiseDistance, DlibLoss, ContrastiveLoss


# Global config
config = Configure()

transform = transforms.Compose([
                         # Scale(96),
                         transforms.Resize((224, 224)),
                         transforms.RandomHorizontalFlip(),
                         transforms.ColorJitter(),
                         transforms.RandomRotation(15),
                         transforms.ToTensor(),
                         transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                                               std = [ 0.5, 0.5, 0.5 ])
                     ])

# Train_dir
# Number of training example: n_batch * n_cls * n_same
train_dir = SiameseWhaleDataset(config.dataroot, config.n_batch, config.n_cls, config.n_same, transform)
train_loader = torch.utils.data.DataLoader(train_dir,
    batch_size=config.n_cls * config.n_same, shuffle=False)

l2_dist = PairwiseDistance(2)

def main():
    global plotter
    plotter = VisdomLinePlotter(env_name=config.visdom_name)
    
    # instantiate model and initialize weights
    model = SiameseNetwork()
    if config.cuda:
        model.cuda()

    optimizer = create_optimizer(model, config.lr)

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            print('=> loading checkpoint {}'.format(config.resume))
            checkpoint = torch.load(config.resume)
            config.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(config.resume))

    start = config.start_epoch
    end = start + config.epochs

    for epoch in range(start, end):
        train(train_loader, model, optimizer, epoch)

def train(train_loader, model, optimizer, epoch):
    model.train()
    # loss_function = DlibLoss()
    loss_function = ContrastiveLoss()
    # pbar = tqdm(enumerate(train_loader))

    for batch_idx, (data_a, data_p, c) in enumerate(train_loader):
        data_a, data_p, c = data_a.cuda(), data_p.cuda(), c.cuda()
        data_a, data_p, c = Variable(data_a), Variable(data_p), Variable(c)

        out_a, out_p = model(data_a), model(data_p)
        loss = loss_function(out_a, out_p, c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer)
        
        plotter.plot('loss', 'train', epoch * config.n_batch + batch_idx, loss.data[0])

        if (epoch * config.n_batch + batch_idx) % config.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data_a),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0]
            ))
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(config.log_dir, epoch))

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = config.lr / (1 + group['step'] * config.lr_decay)

def create_optimizer(model, new_lr):
    # setup optimizer
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=config.wd)
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=config.wd)
    elif config.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=config.lr_decay,
                                  weight_decay=config.wd)
    return optimizer

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='steps',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)

if __name__ == '__main__':
    main()
