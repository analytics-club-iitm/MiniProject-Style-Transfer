import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaptiveInstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine = False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1),1,1)
        gamma, beta = torch.chunk(h, chunks = 2, dim=1)
        return (1+gamma)*self.norm(x) + beta

class AdaResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim = 64, activ = nn.LeakyReLU(0.2), upsample = False):
        super(AdaResBlock, self).__init__()
        self.activ = activ
        self.upsample = upsample
        self.shortcut = dim_in != dim_out

        self.conv_1 = nn.Conv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1)
        self.conv_2 = nn.Conv2d(dim_out, dim_out, kernel_size = 3, stride = 1, padding = 1)
        self.norm_1 = AdaptiveInstanceNorm(style_dim, dim_in)
        self.norm_2 = AdaptiveInstanceNorm(style_dim, dim_out)

        if self.shortcut:
            self.conv_1x1 = nn.Conv2d(dim_in,dim_out, kernel_size = 1, stride = 1, bias=False)

    def _shortcut(self,x):
        if self.upsample:
            x = F.interpolate(x, scale_factor = 2, mode = 'nearest')
        if self.shortcut:
            x = self.conv_1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm_1(x,s)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor = 2, mode = 'nearest')
        x = self.conv_1(x)
        x = self.norm_2(x,s)
        x = self.activ(x)
        x = self.conv_2(x)

        return x
    
    def forward(self, x, s):
        out = self._residual(x,s)
        output = (out + self._shortcut(x))/math.sqrt(2)
        return output

class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim = 64, activ = nn.LeakyReLU(0.2), downsample = False, normalize = False):
        super(ResBlock, self).__init__()
        self.activ = activ
        self.downsample = downsample
        self.shortcut = dim_in != dim_out
        self.normalize = normalize

        self.conv_1 = nn.Conv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1)
        self.conv_2 = nn.Conv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1)
        if self.normalize:
            self.norm_1 = nn.InstanceNorm2d(dim_in, affine = True)
            self.norm_2 = nn.InstanceNorm2d(dim_in, affine = True)

        if self.shortcut:
            self.conv_1x1 = nn.Conv2d(dim_in,dim_out, kernel_size = 1, stride = 1, bias=False)

    def _shortcut(self,x):
        if self.shortcut:
            x = self.conv_1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x,2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm_1(x)
        x = self.activ(x)
        x = self.conv_1(x)
        if self.downsample:
            x = F.avg_pool2d(x,2)
        if self.normalize:
            x = self.norm_2(x)
        x = self.activ(x)
        x = self.conv_2(x)
        return x
    
    def forward(self, x):
        out = self._residual(x)
        output = (out + self._shortcut(x))/math.sqrt(2)
        return output
        
class Discriminator(nn.Module):
    def __init__(self, img_size = 256, num_domains = 2, max_conv_dim = 512):
        super(Discriminator, self).__init__()
        dim_in = 2**14//img_size

        blocks = []

        blocks += [nn.Conv2d(3, dim_in, kernel_size = 3, stride = 1, padding = 1)]

        num_blocks = int(np.log2(img_size)) - 2

        for _ in range(num_blocks):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlock(dim_in, dim_out, downsample = True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, kernel_size = 4, stride = 1)]
        blocks += [nn.LeakyReLU(0.2)]

        # Input - batch_size,1,1,512     Output - batch_size, 1, 1, num_domains
        blocks += [nn.Conv2d(dim_out, num_domains, kernel_size = 1, stride = 1)]

        self.blocks = nn.Sequential(*blocks) 
    
    def forward(self, x, y):
        out = self.blocks(x)
        out = out.view(out.size(0), -1)
        indx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[indx, y]
        return out


class StyleEncoder(nn.Module):
    def __init__(self, img_size = 256, style_dim = 64, num_domains = 2, max_conv_dim = 512):
        super(StyleEncoder, self).__init__()
        dim_in = 2**14//img_size

        blocks = []

        blocks += [nn.Conv2d(3, dim_in, kernel_size = 3, stride = 1, padding = 1)]

        num_blocks = int(np.log2(img_size)) - 2

        for _ in range(num_blocks):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlock(dim_in, dim_out, downsample = True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, kernel_size = 4, stride = 1)]
        blocks += [nn.LeakyReLU(0.2)]

        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]
        
    def forward(self,x,y):
        h = self.shared(x) #y is the domain
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(x)]
        out = torch.stack(out, dim = 1)
        indx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[indx, y]
        return out

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim = 16, style_dim = 64, num_domains = 2):
        super(MappingNetwork, self).__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]

        for _ in range(3):
            layers += [nn.Linear(512,512)]
            layers += [nn.ReLU()]
        
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512,512),
                                            nn.ReLU(),
                                            nn.Linear(512,512),
                                            nn.ReLU(),
                                            nn.Linear(512,512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]
        
    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in unshared:
            out += [layer(h)]
        out = torch.stack(out, dim = 1)
        indx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[indx, y]
        return out

class Generator(nn.Module):
    def __init__(self, img_size = 256, style_dim = 64, max_conv_dim = 512):
        super(Generator, self).__init__()
        dim_in = 2**14//img_size

        self.conv_in = nn.Conv2d(3, dim_in, kernel_size = 3, stride = 1, padding = 1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.conv_out = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine = True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, kernel_size = 1, stride = 1))
        
        num_blocks = int(np.log2(img_size)) - 4

        for _ in range(num_blocks):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encoder.append(ResBlock(dim_in, dim_out, normalize = True, downsample=True))
            self.decoder.insert(0, AdaResBlock(dim_out, dim_in, style_dim, upsample = True))
            dim_in = dim_out

        for _ in range(2):
            self.encoder.append(ResBlock(dim_in, dim_out, normalize = True))
            self.decoder.insert(0, AdaResBlock(dim_out, dim_in, style_dim))
        
    def forward(self, x, s):
        x = self.conv_in(x)
        
        for block in self.encoder:
            x = block(x)
        
        for block in self.decoder:
            x = block(x, s)

        return self.conv_out(x)


if __name__ == "__main__":
    device = torch.device("cpu")
    inp = torch.randn(1, 3, 256, 256).to(device)
    style_code = torch.randn(1, 64).to(device)
    generator = Generator().to(device)
    #mapping = MappingNetwork().to(device)
    #style = StyleEncoder().to
    output = generator(inp, style_code)
    print(output)
