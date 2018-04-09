from dataset import SiameseWhaleDataset
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
from torch.autograd import Variable
from utils import DlibLoss

# For test purpose
features1 = Variable(torch.rand(17 * 7, 128))
features2 = Variable(torch.rand(17 * 7, 128))
label = Variable(torch.rand(17 * 7, ))
# print(label[0])
dlib = DlibLoss()
loss = dlib.forward(features1, features2, label)
print(loss)