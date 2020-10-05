from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
import torch
import starganv2
import facealignment
import os
import numpy as np
import cv2

IMG_SIZE = 256
TRANSFORM = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def load_img(fname):
    img = Image.open(fname).convert('RGB')
    img = TRANSFORM(img)
    return img

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)
"""
helper class for everything
"""
class Model:
    def __init__(self, save_path):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = starganv2.Generator().to(self.device)
        self.mapping_network = starganv2.MappingNetwork().to(self.device)
        self.style_encoder = starganv2.StyleEncoder().to(self.device)
        self.face_mask = facealignment.FaceAlignmentModule().to(self.device)

        # load weights
        self.generator.load_state_dict(torch.load(os.path.join(save_path, "generator.pth")))
        self.mapping_network.load_state_dict(torch.load(os.path.join(save_path, "mapping_network.pth")))
        self.style_encoder.load_state_dict(torch.load(os.path.join(save_path, "style_encoder.pth")))
        self.face_mask.load_state_dict(torch.load(os.path.join(save_path, "facemask.pth")))

    def get_style(self, x, y):
        if len(x.shape) == 3 or len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        if len(x.shape) == 4:
            return torch.squeeze(self.style_encoder(x, y)).cpu().detach()
        elif len(x.shape) == 2:
            return torch.squeeze(self.mapping_network(x, y)).cpu().detach()
        else:
            raise ValueError("Incorect shape (possible latent space/reference image)")

    def apply_style(self, src, style):
        if len(src.shape) == 3:
            src = src.unsqueeze(0)
        if len(style.shape) == 1:
            style = style.unsqueeze(0)
        if len(src.shape) == 4 and len(style.shape) == 2:
            src, style = src.to(self.device), style.to(self.device)
            mask = self.face_mask.get_heatmap(src)
            return torch.squeeze(self.generator(src, style, mask)).cpu().detach()
        else:
            raise ValueError("Incorect image/style code shape")

"""
creating interpolation video as shown in teaser
"""
def create_interpolation_video(model, src_imgs, ref_imgs):
    if not os.path.exists("frames"):
        os.makedirs("frames")

    src = []
    for s in src_imgs:
        src.append(load_img(s).unsqueeze(0))
    src = torch.cat(src, dim=0)
    
    ref = []
    y = []
    for r in ref_imgs:
        ref.append(load_img(r).unsqueeze(0))
        y.append(0) if "female" in r else y.append(1)
    ref = torch.cat(ref, dim=0)
    style_codes = model.get_style(ref, np.array(y))

    alpha = [0] + [sigmoid(alpha) for alpha in np.arange(-5, 5, 0.5) if alpha not in [-1,0,1]] + [1]
    count = 0
    frames = []
    for i in range(len(ref_imgs)-1):
        
        style_now, style_next = style_codes[i], style_codes[i+1]
        ref_now, ref_next = ref[[i]], ref[[i+1]]
        
        for a in alpha:
            style_ref = torch.lerp(style_now, style_next, a)
            out = model.apply_style(src, style_ref.repeat(src.shape[0], 1))
            
            wb = torch.ones(1, 3, 256, 256)
            wb[:,:,int(round(256*(1-a))):256,:] = ref_now[:,:,0:int(round(256*a)),:]
            top_row = torch.cat([wb, src], dim=0)
            
            wb = torch.ones(1, 3, 256, 256)
            wb[:,:,0:int(round(256*(1-a))),:] = ref_now[:,:,int(round(256*a)):256,:]
            wb[:,:,int(round(256*(1-a))):256,:] = ref_next[:,:,0:int(round(256*a)),:]
            bottom_row = torch.cat([wb, out], dim=0)

            f = torch.cat([top_row, bottom_row], dim=0)
            save_image(f, src.shape[0]+1, "frames/{}.png".format(count))
            if a == 0:
                for _ in range(10):
                    frames.append(Image.open("frames/{}.png".format(count)).convert('RGB'))
            frames.append(Image.open("frames/{}.png".format(count)).convert('RGB'))
            count+=1
    
    frames[0].save('out.gif', save_all=True,optimize=False, append_images=frames[1:], duration=100, loop=0)

if __name__ == "__main__":
    model = Model("saved")
    
    import random
    from glob import glob
    
    imgs = sorted(glob(os.path.join("../celeba_hq/val/", "male", '*.jpg'))) + sorted(glob(os.path.join("../celeba_hq/val/", "female", '*.jpg')))
    src_imgs = random.choices(imgs, k=5)
    ref_imgs = random.choices(imgs, k=3)
    
    create_interpolation_video(model, src_imgs, ref_imgs)

    # ref = load_img("../celeba_hq/val/male/000080.jpg")
    # src = load_img("../celeba_hq/val/male/000143.jpg")
    # style = model.get_style(ref, np.array([1]))
    # out = model.apply_style(src, style)
 
    # save_image(src, 1, "src.png")
    # save_image(ref, 1, "ref.png")
    # save_image(out, 1, "out.png")









