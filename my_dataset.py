import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms as transforms
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, cfg, info_csv_path, transform=None):
        super().__init__()
        self.cfg = cfg
        self.transform = transform
        idx_list = [str(i) for i in range(cfg.num_vps)]
        column_names = idx_list + ['mos']
        self.df = pd.read_csv(info_csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.X = self.df[idx_list]
        self.mos = self.df['mos']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_list = []
        for i in range(self.cfg.num_vps):
            if self.cfg.dataset_name == 'JUFE':
                p1, p2, p3, p4 = self.X.iloc[index, i].split("/")
                path = os.path.join(self.cfg.vp_path, p1, p2, p3, p4)
            else:
                p1, p2 = self.X.iloc[index, i].split("/")
                path = os.path.join(self.cfg.vp_path, p1, p2)
            img = Image.open(path)
        
            if self.transform:
                img = self.transform(img)
            img = img.float().unsqueeze(0)
            img_list.append(img)

        imgs = torch.cat(img_list)
        mos = torch.FloatTensor(np.array(self.mos[index]))
        
        return imgs, mos
    
        