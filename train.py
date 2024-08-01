import os
import math
import torch
import torch.nn as nn
from max360iq import Max360IQ as create_model
from utils import train_one_epoch_IQA, test_IQA, compute_max360iq, norm_loss_with_normalization, set_seed
from torch.utils.data import DataLoader
from my_dataset import MyDataset
from config import max360iq_config
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms as transforms
import warnings
from scipy.optimize import OptimizeWarning
import time
import sys


def main(cfg):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(cfg)
    set_seed(cfg)
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    if cfg.use_tensorboard is True:
        ts_path = cfg.tensorboard_path + "/" + cfg.model_name
        if os.path.exists(ts_path) is False:
            os.makedirs(ts_path)
        sw = SummaryWriter(log_dir=ts_path)
    
    # create model
    model = create_model(cfg).to(cfg.device)
    
    compute_max360iq(cfg)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.epochs)) / 2) * (1 - cfg.lrf) + cfg.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    loss_func = norm_loss_with_normalization       # norm-in-norm loss
    
    # load pre-train weight
    if cfg.load_ckpt_path != "":
        assert os.path.exists(cfg.load_ckpt_path), "weights file: '{}' not exist.".format(cfg.load_ckpt_path)
        checkpoint = torch.load(cfg.load_ckpt_path, map_location=cfg.device)
        print(model.load_state_dict(checkpoint['model_state_dict'], strict=False))
        print(checkpoint['model_state_dict'].keys())
        if cfg.continue_training:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("weights had been load!\n")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = MyDataset(cfg=cfg, info_csv_path=cfg.train_info_csv_path, transform=train_transform)
    print(len(train_dataset), "train data has been load!")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, 
        shuffle=True,
        drop_last=True,
    )
    test_dataset = MyDataset(cfg=cfg, info_csv_path=cfg.test_info_csv_path, transform=test_transform)
    print(len(test_dataset), "test data has been load!")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    if cfg.continue_training:
        best_plcc = 0 if cfg.load_ckpt_path == "" else checkpoint['test_plcc']
        begin_epoch = 0 if cfg.load_ckpt_path == "" else checkpoint['epoch']
    else:
        best_plcc = 0
        begin_epoch = 0
    total_time = 0
    print("model:", cfg.model_name, "| dataset:", cfg.dataset_name, "| device:", cfg.device)
    for epoch in range(begin_epoch, cfg.epochs):
        # train
        start_time = time.time()
        train_loss, train_plcc, train_srcc, train_rmse = train_one_epoch_IQA(model, train_loader, loss_func, optimizer, epoch, cfg)
        end_time = time.time()
        spend_time = end_time-start_time
        total_time += spend_time
        print("[train epoch %d/%d] loss: %.6f, plcc: %.4f, srcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min, total time: %.2f h" % \
                (epoch+1, cfg.epochs, train_loss, train_plcc, train_srcc, train_rmse, optimizer.param_groups[0]["lr"], spend_time/60, total_time/3600))
        sys.stdout.flush()
        
        scheduler.step()

        # test
        start_time = time.time()
        test_loss, test_plcc, test_srcc, test_rmse = test_IQA(model, test_loader, loss_func, epoch, cfg)
        end_time = time.time()
        spend_time = end_time-start_time
        total_time += spend_time
        print("[test epoch %d/%d] LOSS: %.6f, PLCC: %.4f, SRCC: %.4f, RMSE: %.4f, LR: %.6f, TIME: %.2f MIN, total time: %.2f h" % \
                (epoch+1, cfg.epochs, test_loss, test_plcc, test_srcc, test_rmse, optimizer.param_groups[0]["lr"], spend_time/60, total_time/3600))
        sys.stdout.flush()
        
        if cfg.use_tensorboard is True:
            sw.add_scalars(cfg.model_name+"/"+cfg.dataset_name+" Loss", {'train': train_loss, 'test': test_loss}, epoch)
            sw.add_scalars(cfg.model_name+"/"+cfg.dataset_name+" plcc", {'train': train_plcc, 'test': test_plcc}, epoch)
            sw.add_scalars(cfg.model_name+"/"+cfg.dataset_name+" srcc", {'train': train_srcc, 'test': test_srcc}, epoch)
            sw.add_scalars(cfg.model_name+"/"+cfg.dataset_name+" rmse", {'train': train_rmse, 'test': test_rmse}, epoch)
            sw.add_scalar(cfg.model_name+"/"+cfg.dataset_name+" learning_rate", optimizer.param_groups[0]["lr"], epoch)
        
        if test_plcc > best_plcc:
            best_plcc = test_plcc
            w_phat = cfg.save_ckpt_path + "/" + cfg.model_name
            if os.path.exists(w_phat) is False:
                os.makedirs(w_phat)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, w_phat + "/best_epoch_"+str(epoch+1)+".pth")

if __name__ == '__main__':
    cfg = max360iq_config(dataset='OIQA')
    main(cfg)