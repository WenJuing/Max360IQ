import torch
import torch.nn as nn
import torch.nn.functional as F
from maxvit_backbone import MaxViTStage, ConvBlock
import numpy as np


class Max360IQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_vps = cfg.num_vps
        self.use_gru = cfg.use_gru
        self.stem = Stem(cfg)
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, sum(cfg.depths))]
        stages = []
        for index, (depth, channel) in enumerate(zip(cfg.depths, cfg.channels)):
            stage = MaxViTStage(
                num_blocks=depth,
                in_channels=cfg.dim if index==0 else cfg.channels[index-1],
                out_channels=channel,
                num_heads=cfg.num_heads[index],
                drop_path=dpr[sum(cfg.depths[:index]):sum(cfg.depths[:index + 1])],
                cfg=cfg
            )
            stages.append(stage)
        self.stages = nn.ModuleList(stages)
        self.gem_pooling = GeM()
        self.integrate_feat = FeatureIntegration()
        self.reduce = nn.Linear(np.sum(cfg.channels), cfg.channels[-1], bias=False)
        self.gru = nn.GRU(2*cfg.channels[-1], cfg.gru_hidden_dim, cfg.gru_layer_dim, batch_first=True)
        self.regression = Regression(cfg)
        
        self.apply(_weight_init)

    def vp_forward(self, x):
        x = self.stem(x)
        multi_scale_feats = []
        for stage in self.stages:
            x = stage(x)
            multi_scale_feats.append(x)
        guidance_vector = self.gem_pooling(x).unsqueeze(1)
        x = self.reduce(self.integrate_feat(multi_scale_feats).unsqueeze(1))
        x = torch.cat((x, guidance_vector), dim=-1)
        return x
    
    def forward(self, x):
        xs = []
        for i in range(x.shape[1]):
            xs.append(self.vp_forward(x[:, i, ...]))
            
        if self.use_gru:
            x, _ = self.gru(torch.cat(xs, dim=1), None)
        else:
            x = torch.cat(xs, dim=1)
        pred = self.regression(x)

        return pred
    
    
class Stem(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = ConvBlock(cfg.img_channels, cfg.dim, 3, stride=2, padding=1)
        self.conv2 = ConvBlock(cfg.dim, cfg.dim, 3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x
    

class Regression(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.quality = nn.Sequential(
            nn.Linear(2*cfg.channels[-1], cfg.reg_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.reg_hidden_dim, 1, bias=False)
        )
    
    def forward(self, x):
        x = self.quality(x)
        return x.mean(dim=1).squeeze(dim=-1)

        
class FeatureIntegration(nn.Module):
    def __init__(self):
        super().__init__()
        self.gem1 = GeM()
        self.gem2 = GeM()
        self.gem3 = GeM()
        self.gem4 = GeM()
    
    def forward(self, xs):
        x1 = self.gem1(xs[0])
        x2 = self.gem1(xs[1])
        x3 = self.gem1(xs[2])
        x4 = self.gem1(xs[3])

        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x
        
        
        
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad=True):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.requires_grad = requires_grad

    def forward(self, x):
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p).view(x.size(0),-1)

        return x
    

def _weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
