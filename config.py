import torch
import os

class Configure(object):
    def __init__(self):
        super(Configure, self).__init__()
        self.dataroot = "../../input/dlib_gen"
        self.n_cls = 10
        self.n_same = 5
        self.n_batch = 1000# Number of training example: n_batch * n_cls * n_same
        self.lr = 0.1
        self.lr_decay = 1e-4
        self.wd = 0
        self.optimizer = 'sgd'
        self.resume = None# "logs/checkpoint_0.pth"
        self.start_epoch = 0
        self.epochs = 5
        self.cuda = torch.cuda.is_available()
        self.log_interval = 10
        self.visdom_name = "siamese_whale"
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
