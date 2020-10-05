import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

"""
an alternative for batch norm and instance norm
essentially takes in input feature x and style image y 
and aligns the mean and variance of x to match y. 
has no learnable affline parameters instead it computes them 
adaptively from the style input
"""
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaptiveInstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s):
        # generate params gamma and beta
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1+gamma)*self.norm(x) + beta

class AdaResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, filter_wt=0, activ=nn.LeakyReLU(0.2), upsample=False):
        super(AdaResBlock, self).__init__()
        self.filter_wt = filter_wt
        self.activ = activ
        self.upsample = upsample
        self.shortcut = dim_in != dim_out

        self.conv_1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        self.norm_1 = AdaptiveInstanceNorm(style_dim, dim_in)
        self.norm_2 = AdaptiveInstanceNorm(style_dim, dim_out)
        if self.shortcut:
            self.conv_1x1 = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.shortcut:
            x = self.conv_1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm_1(x, s)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv_1(x)
        x = self.norm_2(x, s)
        x = self.activ(x)
        x = self.conv_2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.filter_wt == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activ=nn.LeakyReLU(0.2), normalize=False, downsample=False):
        super(ResBlock, self).__init__()
        self.activ = activ
        self.normalize = normalize
        self.downsample = downsample
        self.shortcut = dim_in != dim_out
        self.conv_1 = nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        if self.normalize:
            self.norm_1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm_2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.shortcut:
            self.conv_1x1 = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False)

    def _shortcut(self, x):
        if self.shortcut:
            x = self.conv_1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm_1(x)
        x = self.activ(x)
        x = self.conv_1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm_2(x)
        x = self.activ(x)
        x = self.conv_2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super(Discriminator, self).__init__()
        dim_in = 2**14//img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, kernel_size=3, stride=1, padding=1)]
        num_blocks = int(np.log2(img_size)) - 2
        for _ in range(num_blocks):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlock(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, kernel_size=4, stride=1)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, kernel_size=1, stride=1)]
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x, y):
        out = self.blocks(x) # y denotes the domain
        out = out.view(out.size(0), -1)
        idx = [ _ for _ in range(len(y))]
        out = out[idx, y]
        return out

"""
Given an image x and its corresponding domain y, 
the style encoder extracts the style code
"""
class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super(StyleEncoder, self).__init__()
        dim_in = 2**14//img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, kernel_size=3, stride=1, padding=1)]
        num_blocks = int(np.log2(img_size)) - 2
        for _ in range(num_blocks):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlock(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]
    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)
        idx = [ _ for _ in range(len(y))]
        s = out[idx, y]
        return s

"""
Given a latent code z and a domain y, 
the mapping network generates a style code
"""
class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super(MappingNetwork, self).__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)
        idx = [ _ for _ in range(len(y))]
        s = out[idx, y]
        return s

"""
if one pixel is brighter than the ones around it then it gets boosted more. High pass filter which can help 
accentuate the boundaries. 
"""
class HighPass(nn.Module):
    def __init__(self, filter_wt):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]) / filter_wt

    def forward(self, x):
        filter = (self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)).to(x.device)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, filter_wt=1):
        super(Generator, self).__init__()
        dim_in = 2**14 // img_size
        self.conv_in = nn.Conv2d(3, dim_in, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.conv_out = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, kernel_size=1, stride=1))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if filter_wt > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encoder.append(ResBlock(dim_in, dim_out, normalize=True, downsample=True))
            self.decoder.insert(0, AdaResBlock(dim_out, dim_in, style_dim, filter_wt=filter_wt, upsample=True))
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encoder.append(ResBlock(dim_out, dim_out, normalize=True))
            self.decoder.insert(0, AdaResBlock(dim_out, dim_out, style_dim, filter_wt=filter_wt))

        if filter_wt > 0:
            self.highpass_filter = HighPass(filter_wt)

    def forward(self, x, s, masks=None):
        x = self.conv_in(x)
        cache = {}
        for block in self.encoder:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
        for block in self.decoder:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.highpass_filter(mask * cache[x.size(2)]) # add skip connections
        return self.conv_out(x)