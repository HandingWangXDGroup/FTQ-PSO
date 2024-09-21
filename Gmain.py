#    Authors:    Chao Li, Tingsong Jiang，Handing Wang, Wen Yao, Donghua Wang
#    Xidian University, China
#    Defense Innovation Institute, Chinese Academy of Military Science, China
#    Zhongyuan University of Technology, China
#    EMAIL:      lichaoedu@126.com
#    DATE:       September 2024
# ------------------------------------------------------------------------
# This code is part of the program that produces the results in the following paper:
#
# Chao Li, Tingsong Jiang，Handing Wang, Wen Yao, Donghua Wang, Optimizing Latent Variables in Integrating Transfer and Query Based Attack Framework, IEEE Transactions on  Pattern Analysis and Machine Intelligence, 2024.
#
# You are free to use it for non-commercial purposes. However, we do not offer any forms of guanrantee or warranty associated with the code. We would appreciate your acknowledgement.
# ------------------------------------------------------------------------
import torch
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from psogan import psog
import os
import warnings

warnings.filterwarnings("ignore")

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset():
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        data_root = r'\root'
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            words[0] = os.path.join(data_root, words[0])
            assert os.path.exists(words[0])
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def backward(self):
        return "mean={}, std={}".format(self.mean, self.std)


def normalize_fn(tensor, mean, std):

    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)





root = " "   #the root of the dataset
test_data = MyDataset(txt=root + 'val.txt', transform=transforms.Compose([
                                                  transforms.Resize((224, 224)),
                                                  transforms.ToTensor()
                                                    ]))


test_loader = DataLoader(dataset=test_data, batch_size=1,shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = models.googlenet(pretrained=True)

normalize = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
model = torch.nn.Sequential(normalize, model)
model = model.to(device)
model.eval()

attack_success = 0
total_count = 0
net_correct = 0
total_norm = 0

for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, pre = torch.max(outputs.data, 1)
      total_count += 1
      if pre == labels:
          net_correct += 1
          adv_images, FES, p_norm = psog(images, labels, model, device)

          total_norm += p_norm
          adv_images = adv_images.to(device)
          adv_outputs = model(adv_images)
          _, adv_pre = torch.max(adv_outputs.data, 1)
          if adv_pre != labels:
             attack_success += 1

      if net_correct > 0:
          print('Ratio of attack success: %f %%' % (100 * float(attack_success) / net_correct))

      if net_correct == 1000:
         break