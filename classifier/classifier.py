from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from torchvision import transforms, models
import os
from glob import glob
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

"""
Basic transforms to get images ready for resnet model
"""
IMG_SIZE = 256
TRANSFORM = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


"""
dataset class for returning labels and corresponding image file
"""
class celeba(Dataset):
    def __init__(self, img_files, labels):
        self.img_files = img_files
        self.labels = labels

    def __getitem__(self, indx):
        img = Image.open(self.img_files[indx])
        img = TRANSFORM(img)
        label = torch.Tensor(self.labels[indx])
        return img, label

    def __len__(self):
        return len(self.img_files)

img_files = sorted(glob(os.path.join("img_align_celeba", '*.jpg')))

label_file = open("list_attr_celeba.txt").readlines()[2:]
labels = [label_file[i].split() for i in range(len(label_file))]
for i in range(len(labels)):
    labels[i] = [int(n.replace('-1', '0')) for n in labels[i][1:]]

print(open("list_attr_celeba.txt").readlines()[1].split())

"""
Resnet pretrained model with linear layers in the end.
"""
class Model(nn.Module):
    def __init__(self, classes):
        super(Model, self).__init__()
        resnet = models.resnet34(pretrained=False)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, classes)
        )
    def forward(self, x):
        x = self.resnet_backbone(x)
        x = self.fc(x.view(-1, 512))
        x = torch.sigmoid(x)
        return x

model = Model(len(labels[0]))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

img_train, img_test, label_train, label_test = train_test_split(img_files, labels, test_size=0.3, random_state=42)

train_dataset = celeba(img_train, label_train)
test_dataset = celeba(img_test, label_test)
train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=8)
"""
since the labels are 0 or 1 based on presence or absence of an attribute we use the binary cross entropy loss
"""
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.999))
best = 0

for epoch in range(100):
    pbar = tqdm(total=len(train_dataloader), desc="Epoch: {}".format(epoch))
    correct = []
    for step, data in enumerate(train_dataloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        pred_choice = outputs > 0.5
        # we divide by 40 as we are counting each individual correct attribute in the accuracy as well.
        c = (pred_choice==labels).cpu().sum().item()/(imgs.shape[0]*40)
        correct.append(c)
        pbar.set_description("train epoch: {}, accuracy: {:.4f}".format(epoch, np.mean(correct)))
        pbar.update(1)

    if epoch%5 == 0:
        pbar = tqdm(total=len(test_dataloader), desc="Epoch: {}".format(epoch))
        correct = []
        for step, data in enumerate(test_dataloader):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs)

            pred_choice = outputs > 0.5
            c = (pred_choice==labels).cpu().sum().item()/(imgs.shape[0]*40)
            correct.append(c)
            pbar.set_description("test epoch: {}, accuracy: {:.4f}".format(epoch, np.mean(correct)))
            pbar.update(1)

        if np.mean(correct) > best:
            torch.save(model.state_dict(), "classifier_best.pth")

    torch.save(model.state_dict(), "classifier_last.pth")






    