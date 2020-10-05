import infer
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from glob import glob
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split

"""
generating style codes and saving in corresponding domain.csv files
"""
def gen_style_codes():
    male_imgs = sorted(glob(os.path.join("../celeba_hq/val/", "male", '*.jpg'))) + sorted(glob(os.path.join("../celeba_hq/train/", "male", '*.jpg')))
    female_imgs = sorted(glob(os.path.join("../celeba_hq/val/", "female", '*.jpg'))) + sorted(glob(os.path.join("../celeba_hq/train/", "female", '*.jpg')))

    model = infer.Model("saved")
    id = []
    pbar = tqdm(total=len(male_imgs))
    style_codes = []
    for img_file in male_imgs:
        img = infer.load_img(img_file)
        style = model.get_style(img, np.array([1]))
        style_codes.append(style.numpy())
        id.append(os.path.basename(img_file).split(".")[0])
        pbar.update(1)
    pbar.close()
    male_csv = pd.DataFrame(np.array(style_codes))
    male_csv["id"] = id
    male_csv.to_csv("male.csv", index=False)

    id = []
    style_codes = []
    pbar = tqdm(total=len(female_imgs))
    for img_file in female_imgs:
        img = infer.load_img(img_file)
        style = model.get_style(img, np.array([0]))
        style_codes.append(style.numpy())
        id.append(os.path.basename(img_file).split(".")[0])
        pbar.update(1)
    pbar.close()
    female_csv = pd.DataFrame(np.array(style_codes))
    female_csv["id"] = id
    female_csv.to_csv("female.csv", index=False)

"""
plot and see
"""
def plot_latent_space():
    pca = PCA(3)

    male_csv = pd.read_csv("male.csv")
    male_style_codes = np.array(male_csv.iloc[:,0:64])
    classes = [1 for i in range(len(male_style_codes))]
    female_csv = pd.read_csv("female.csv")
    female_style_codes = np.array(female_csv.iloc[:,0:64])
    classes += [0 for i in range(len(female_style_codes))]
    
    style_codes = np.concatenate([male_style_codes, female_style_codes], axis=0)
    style_codes_3d = pca.fit_transform(style_codes)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(style_codes_3d[:,0], style_codes_3d[:,1], style_codes_3d[:,2], c=classes)
    plt.show()

"""
classifier model resnet
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

"""
classify images based on attributes and save in domain.csv files
"""
all_attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
def classify_attributes():
    model = Model(len(all_attributes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load("classifier/classifier_best.pth"))
    model.eval()

    male_imgs = sorted(glob(os.path.join("../celeba_hq/val/", "male", '*.jpg'))) + sorted(glob(os.path.join("../celeba_hq/train/", "male", '*.jpg')))
    female_imgs = sorted(glob(os.path.join("../celeba_hq/val/", "female", '*.jpg'))) + sorted(glob(os.path.join("../celeba_hq/train/", "female", '*.jpg')))
    
    IMG_SIZE = 256
    TRANSFORM = transforms.Compose([
        transforms.Resize([IMG_SIZE, IMG_SIZE]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    male_csv = pd.read_csv("male.csv")
    pbar = tqdm(total=len(male_imgs))
    male_attributes = []
    for img_file in male_imgs:
        img = Image.open(img_file)
        img = TRANSFORM(img)
        with torch.no_grad():
            out =  model(img.unsqueeze(0).to(device))
        male_attributes.append((torch.squeeze(out)>0.5).cpu().detach().numpy())
        pbar.update(1)
    male_attributes = np.array(male_attributes).astype("int")
    for i in range(len(all_attributes)):
        male_csv[all_attributes[i]] = male_attributes[:,i]
    male_csv.to_csv("male.csv", index=False)
    pbar.close()

    female_csv = pd.read_csv("female.csv")
    pbar = tqdm(total=len(female_imgs))
    female_attributes = []
    for img_file in female_imgs:
        img = Image.open(img_file)
        img = TRANSFORM(img)
        with torch.no_grad():
            out =  model(img.unsqueeze(0).to(device))
        female_attributes.append((torch.squeeze(out)>0.5).cpu().detach().numpy())
        pbar.update(1)
    female_attributes = np.array(female_attributes).astype("int")
    for i in range(len(all_attributes)):
        female_csv[all_attributes[i]] = female_attributes[:,i]
    female_csv.to_csv("female.csv", index=False)
    pbar.close()

"""
train svms to identify boundary for each attribute
"""        
def train_svms():
    req_attributes = ['Black_Hair', 'Blond_Hair', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Lipstick', 'Young', 'Attractive']
    domain = "female"
    csv = pd.read_csv("{}.csv".format(domain))
    x,y = np.array(csv.iloc[:,0:64]), np.array(csv.iloc[:, 65:65+len(all_attributes)])
    for req in req_attributes:
        true_indx = y[:,all_attributes.index(req)] == 1
        neg_indx = y[:,all_attributes.index(req)] == 0
        print("Attribute: {} True: {} Negative: {}".format(req, np.sum(true_indx), np.sum(neg_indx)))
        
        clf = svm.SVC(kernel='linear')
        classifier = clf.fit(x, y[:,all_attributes.index(req)])
        coeffs = classifier.coef_.reshape(1, x.shape[1]).astype(np.float32)  
        np.save(open(f"{domain}_{req}.npy", "wb"), coeffs/np.linalg.norm(coeffs))

if __name__ == "__main__": 
    # gen_style_codes()
    # plot_latent_space()
    # classify_attributes()
    train_svms()