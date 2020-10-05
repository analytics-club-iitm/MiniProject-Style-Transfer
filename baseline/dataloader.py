from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import os
import random
from PIL import Image
import torch

"""
CelebA dataset class inheriting modules from torch dataset class. 
Must contain the following methods: __init__(), __getitem__(), __len__()
"""
class CelebA(Dataset):
    """
    our init method: sets up the class variables.
    populate labels and img paths 
    """
    def __init__(self, root_dir, config, transform=False):
        self.root_dir = root_dir
        self.config = config
        self._init_dataset()
        if transform:
            self._init_transform()
    
    def _init_dataset(self):
        domains = os.listdir(self.root_dir)
        fnames_1, fnames_2, labels = [], [], []

        for indx, domain in enumerate(sorted(domains)):
            img_files = sorted(glob(os.path.join(self.root_dir, domain, '*.jpg')))
            fnames_1 += img_files
            fnames_2 += random.sample(img_files, len(img_files))
            labels += [indx]*len(img_files)
        
        self.src_imgs = fnames_1
        self.ref_imgs = list(zip(fnames_1, fnames_2))
        self.src_labels = labels
        self.ref_labels = labels
        self._shuffle()
    
    """
    this method is called to shuffle the dataset and the corresponding labels.
    """
    def _shuffle(self):
        temp = list(zip(self.src_imgs, self.src_labels))
        random.shuffle(temp)
        self.src_imgs, self.src_labels = zip(*temp)
        self.src_imgs = list(self.src_imgs)
        self.src_labels = list(self.src_labels)

        temp = list(zip(self.ref_imgs, self.ref_labels))
        random.shuffle(temp)
        self.ref_imgs, self.ref_labels = zip(*temp)
        self.ref_imgs = list(self.ref_imgs)
        self.ref_labels = list(self.ref_labels)
    
    """
    initialising the transformation performed on the images.
    randomly crop with a probability
    resize images to fixed size
    randomly flip images horizontally
    convert PIL image to tensors
    normalise image tensors to 0 mean and unit variance
    """
    def _init_transform(self):
        crop = transforms.RandomResizedCrop(self.config["img_size"], scale=[0.8,1.0], ratio=[0.9,1.1])
        rand_crop = transforms.Lambda(lambda x: crop(x) if random.random()<self.config["prob"] else x)
        self.transform = transforms.Compose([
            rand_crop,
            transforms.Resize([self.config["img_size"], self.config["img_size"]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    """
    used to return input and target tensors
    returns items are: (src image, src label), (ref image, ref label, latent vec), (ref image, ref label, latent vec) 
    basically source image, labels and two reference images with corresponding labels and latent vectors.
    """
    def __getitem__(self, index):

        src = self.src_imgs[index]
        ref1, ref2 = self.ref_imgs[index]
        src_label = self.src_labels[index]
        ref_label = self.ref_labels[index]

        src = Image.open(src).convert('RGB')
        ref1 = Image.open(ref1).convert('RGB')
        ref2 = Image.open(ref2).convert('RGB')
        if self.transform is not None:
            src = self.transform(src)
            ref1 = self.transform(ref1)
            ref2 = self.transform(ref2)
        
        src_label = torch.tensor(src_label, dtype=torch.long)
        ref_label = torch.tensor(ref_label, dtype=torch.long)

        latent1 = torch.randn(self.config["latent_dim"])
        latent2 = torch.randn(self.config["latent_dim"])

        return src, src_label, ref1, ref2, ref_label, latent1, latent2

    """
    returns the length of the dataset
    """
    def __len__(self):
        return len(self.src_imgs)










        