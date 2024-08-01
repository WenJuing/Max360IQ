import torch
from max360iq import Max360IQ as create_model
from utils import mean_squared_error, logistic_func, fit_function, set_seed, mean_mos
from torch.utils.data import DataLoader
from my_dataset import MyDataset
from config import max360iq_config
from torchvision import transforms as transforms
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm


def main(cfg):
    set_seed(cfg)
    print("*****begin test*******************************************************")
    model = create_model(cfg).cuda(cfg.device)
    checkpoint = torch.load(cfg.load_ckpt_path, map_location=cfg.device)
    print(model.load_state_dict(checkpoint['model_state_dict'], strict=False))
    print("weights had been load!\n")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
    ])

    test_dataset = MyDataset(cfg=cfg, info_csv_path=cfg.test_info_csv_path, transform=test_transform)
    print(len(test_dataset), "test data has been load!")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    
    # test
    model.eval()
    pred_all = []
    mos_all = []
    with torch.no_grad():
        test_loader = tqdm(test_loader) 
        for i, (img, mos) in enumerate(test_loader):
            pred = model(img.to(cfg.device))
            
            pred_all = np.append(pred_all, pred.cpu().data.numpy())
            mos_all = np.append(mos_all, mos.data.numpy())
            
            test_loader.set_description("[test epoch %d]" % (checkpoint['epoch']+1))


    mos_all, pred_all = mean_mos(pred_all, mos_all, cfg.num_sequence)
    logistic_pred_all = fit_function(mos_all, pred_all)
    plcc = pearsonr(logistic_pred_all, mos_all)[0]
    srcc = spearmanr(logistic_pred_all, mos_all)[0]
    rmse = mean_squared_error(logistic_pred_all, mos_all, squared=False) 
    print("plcc: %.4f, srcc: %.4f, rmse: %.4f" % (plcc, srcc, rmse))

if __name__ == '__main__':
    cfg = max360iq_config(dataset='OIQA')
    cfg['load_ckpt_path'] = "/home/fang/tzw1/ckpt/paper/oiqa_best_epoch_72.pth"
    cfg['vp_path'] = '/media/fang/Elements/datasets/OIQA/viewports_8'
    cfg['test_info_csv_path'] = "/home/fang/tzw1/databases/OIQA_vp2_test_info.csv"
    main(cfg)