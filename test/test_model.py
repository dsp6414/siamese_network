from dataset import SiameseWhaleDataset
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
from torch.autograd import Variable
from model import EmbeddingLayer

# For testing purpose
x = Variable(torch.FloatTensor(8, 3, 224, 224))
embedding_layer = EmbeddingLayer()
o = embedding_layer(x)
print("Output shape {}".format(o.shape))