from dataset import SiameseWhaleDataset
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
from torch.autograd import Variable

n_cls = 17
n_same = 7
dataroot = "../../../input/dlib_gen"

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

train_dir = SiameseWhaleDataset(dataroot, n_cls, n_same, transform)
train_loader = torch.utils.data.DataLoader(train_dir,
    batch_size=n_cls * n_same, shuffle=False)

pbar = tqdm(enumerate(train_loader))

for batch_idx, (data_a, data_p, c) in pbar:
    data_a, data_p, c = Variable(data_a), Variable(data_p), Variable(c)
    print("Shape data_a {}".format(data_a.shape))
    print("Shape data_p {}".format(data_p.shape))
    print("Shape c {}".format(c.shape))

print(c)