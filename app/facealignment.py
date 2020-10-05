import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from functools import partial

index_map = {
    "chin": {"start":8,"end":25},
    "eyebrows": {"start":33,"end":51},
    "eyebrowsedges": {"start":33,"end":46},
    "nose": {"start":51,"end":55},
    "nostrils": {"start":55,"end":60},
    "eyes": {"start":60,"end":76},
    "lipedges": {"start":76,"end":82},
    "lipupper": {"start":77,"end":82},
    "liplower": {"start":83,"end":88},
    "lipinner": {"start":88,"end":96}
}

def resize(x, p=2):
    return x**p

def shift(x, N):
    up = N>0
    N = abs(N)
    _,_, H, W = x.size()
    head = torch.arange(N)
    tail = torch.arange(H-N)

    if up: 
        head = torch.arange(H-N) + N
        tail = torch.arange(N)
    else:
        head = torch.arange(N) + (H-N)
        tail = torch.arange(H-N)
    
    perm = torch.cat([head, tail]).to(x.device)
    out = x[:,:,perm,:]
    return out

def preprocess(x):
    N,C,H,W = x.size()
    x = torch.where(x<0.1, torch.zeros_like(x), x)
    x = x.contiguous()
    
    x_ = x.view(N*C, -1)
    max_val = torch.max(x_, dim=1, keepdim=True)[0]
    min_val = torch.min(x_, dim=1, keepdim=True)[0]
    x_ = (x_ - min_val)/(max_val-min_val+1e-6)
    x = x_.view(N,C,H,W)

    sw = H//256
    operations = {
        "chin": {"shift":0, "resize":3},
        "eyebrows": {"shift":-7*sw, "resize":2},
        "nostrils": {"shift":8*sw, "resize":4},
        "lipupper": {"shift":-8*sw, "resize":4},
        "liplower": {"shift":8*sw, "resize":4},
        "lipinner": {"shift":-2*sw, "resize":3}
    }

    for part, ops in operations.items():
        start, end = index_map[part].values()
        x[:,start:end] = resize(shift(x[:, start:end], ops["shift"]), ops["resize"])
    
    zero_out = torch.cat([
        torch.arange(0, index_map["chin"]["start"]),
        torch.arange(index_map["chin"]["end"], 33),
        torch.LongTensor([index_map["eyebrowsedges"]["start"], index_map["eyebrowsedges"]["end"], index_map["lipedges"]["start"], index_map["lipedges"]["end"]])
    ])

    x[:, zero_out] = 0

    start, end = index_map["nose"].values()
    x[:, start+1:end] = shift(x[:, start+1:end], 4*sw)
    x[:, start:end] = resize(x[:, start:end], 1)

    start, end = index_map["eyes"].values()
    x[:, start:end] = resize(x[:, start:end], 1)
    x[:, start:end] = resize(shift(x[:, start:end], -8), 3) + shift(x[:, start:end], -24)

    x_ = deepcopy(x)
    x_[:, index_map["chin"]["start"]:index_map["chin"]["end"]] = 0
    x_[:, index_map["lipedges"]["start"]:index_map["lipinner"]["end"]] = 0
    x_[:, index_map["eyebrows"]["start"]:index_map["eyebrows"]["end"]] = 0

    x = torch.sum(x, dim=1, keepdim=True)
    x_ = torch.sum(x_, dim=1, keepdim=True)

    x[x != x] = 0
    x_[x != x] = 0
    return x.clamp_(0, 1), x_.clamp_(0, 1)

class AddCoordinates(nn.Module):
    def __init__(self, height=64, width=64, with_r=False, with_boundary=False):
        super(AddCoordinates, self).__init__()
        self.with_r = with_r
        self.with_boundary = with_boundary
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            x_coords = torch.arange(height).unsqueeze(1).expand(height, width).float()
            y_coords = torch.arange(width).unsqueeze(0).expand(height, width).float()
            x_coords = (x_coords / (height - 1)) * 2 - 1
            y_coords = (y_coords / (width - 1)) * 2 - 1
            coords = torch.stack([x_coords, y_coords], dim=0)

            if self.with_r:
                rr = torch.sqrt(torch.pow(x_coords, 2) + torch.pow(y_coords, 2))
                rr = (rr / torch.max(rr)).unsqueeze(0)
                coords = torch.cat([coords, rr], dim=0)

            self.coords = coords.unsqueeze(0).to(device)
            self.x_coords = x_coords.to(device)
            self.y_coords = y_coords.to(device)

    def forward(self, x, heatmap=None):
        coords = self.coords.repeat(x.size(0), 1, 1, 1)

        if self.with_boundary and heatmap is not None:
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)
            zero_tensor = torch.zeros_like(self.x_coords)
            xx_boundary_channel = torch.where(boundary_channel > 0.05, self.x_coords, zero_tensor).to(zero_tensor.device)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, self.y_coords, zero_tensor).to(zero_tensor.device)
            coords = torch.cat([coords, xx_boundary_channel, yy_boundary_channel], dim=1)

        x_and_coords = torch.cat([x, coords], dim=1)
        return x_and_coords


class CoordConv(nn.Module):
    def __init__(self, height, width, with_r, with_boundary, in_channels, first_one=False, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoordinates(height, width, with_r, with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, x, heatmap=None):
        ret = self.addcoords(x, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, int(out_channels / 2), kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.bn3 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)

        self.downsample = None

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
            )

    def forward(self, x):
        residual = x

        out1 = self.conv1(F.relu(self.bn1(x), True))
        out2 = self.conv2(F.relu(self.bn2(out1), True))
        out3 = self.conv3(F.relu(self.bn3(out2), True))

        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual
        return out3

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConv(64, 64, True, True, 256, first_one,
                                     out_channels=256,
                                     kernel_size=1, stride=1, padding=0)
        self._create_block(self.depth)

    def _create_block(self, depth):
        self.add_module('b1_' + str(depth), ConvBlock(256, 256))
        self.add_module('b2_' + str(depth), ConvBlock(256, 256))
        if depth > 1:
            self._create_block(depth - 1)
        else:
            self.add_module('b2_plus_' + str(depth), ConvBlock(256, 256))
        self.add_module('b3_' + str(depth), ConvBlock(256, 256))

    def _forward_block(self, x, level):
        up1 = x
        up1 = self._modules['b1_' + str(level)](up1)
        low1 = F.avg_pool2d(x, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward_block(low1, level - 1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)
        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x, heatmap):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward_block(x, self.depth), last_channel

def get_preds_fromhm(hm):
    max, idx = torch.max(hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor([hm_[pY, pX + 1] - hm_[pY, pX - 1], hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)
    return preds

class FaceAlignmentModule(nn.Module):
    def __init__(self, num_modules=1, end_relu=False, num_landmarks=98):
        super(FaceAlignmentModule, self).__init__()
        self.num_modules = num_modules
        self.end_relu = end_relu

        self.conv1 = CoordConv(256, 256, True, False, in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        self.add_module('m0', HourGlass(1, 4, 256, first_one=True))
        self.add_module('top_m_0', ConvBlock(256, 256))
        self.add_module('conv_last0', nn.Conv2d(256, 256, 1, 1, 0))
        self.add_module('bn_end0', nn.BatchNorm2d(256))
        self.add_module('l0', nn.Conv2d(256, num_landmarks+1, 1, 1, 0))

    def load_pretrained_weights(self, fname):
        if torch.cuda.is_available():
            checkpoint = torch.load(fname)
        else:
            checkpoint = torch.load(fname, map_location=torch.device('cpu'))
        model_weights = self.state_dict()
        model_weights.update({k: v for k, v in checkpoint['state_dict'].items() if k in model_weights})
        self.load_state_dict(model_weights)

    def forward(self, x):
        x, _ = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        outputs = []
        boundary_channels = []
        tmp_out = None
        ll, boundary_channel = self._modules['m0'](x, tmp_out)
        ll = self._modules['top_m_0'](ll)
        ll = F.relu(self._modules['bn_end0'](self._modules['conv_last0'](ll)), True)

        tmp_out = self._modules['l0'](ll)
        if self.end_relu:
            tmp_out = F.relu(tmp_out)
        outputs.append(tmp_out)
        boundary_channels.append(boundary_channel)
        return outputs, boundary_channels
    
    @torch.no_grad()
    def get_heatmap(self, x, b_preprocess=True):
        x = F.interpolate(x, size=256, mode='bilinear')
        x_01 = x*0.5 + 0.5
        outputs, _ = self(x_01)
        heatmaps = outputs[-1][:, :-1, :, :]
        scale_factor = x.size(2) // heatmaps.size(2)
        if b_preprocess:
            heatmaps = F.interpolate(heatmaps, scale_factor=scale_factor, mode='bilinear', align_corners=True)
            heatmaps = preprocess(heatmaps)
        return heatmaps
    
    @torch.no_grad()
    def get_landmark(self, x):
        heatmaps = self.get_heatmap(x, b_preprocess=False)
        landmarks = []
        for i in range(x.size(0)):
            pred_landmarks = get_preds_fromhm(heatmaps[i].cpu().unsqueeze(0))
            landmarks.append(pred_landmarks)
        scale_factor = x.size(2) // heatmaps.size(2)
        landmarks = torch.cat(landmarks) * scale_factor
        return landmarks

if __name__ == "__main__":
    facemask = FaceAlignmentModule().cuda()
    facemask.load_state_dict(torch.load("saved/facemask.pth"))

    from PIL import Image
    src = "../celeba_hq/train/male/000016.jpg"
    src = Image.open(src).convert('RGB')

    from torchvision import transforms
    img_size = 256
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    src = transform(src).cuda()
    import torchvision.utils as vutils
    def denormalize(x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def save_image(x, ncol, filename):
        x = denormalize(x)
        vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

    src = src.unsqueeze(0)
    with torch.no_grad():
        x1, x2 = facemask.get_heatmap(src)

    save_image(src, 1, "img.png")
    save_image(src*x1, 1, "img_x1.png")
    save_image(src*x2, 1, "img_x2.png")