import torch


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def max360iq_config(dataset='CVIQ'):
    config = Config({
        # model setting
        'num_vps': 1,                       # number of viewports in a sequence.
        'num_sequence': 8,                  # number of sequence of OI.
        'img_channels': 3,
        'img_size': 224,
        'dim': 64,                          # dimension after stem module.
        'depths': (2, 2, 3, 2),             # number of maxvit block in each stage.
        'channels': (64, 64, 128, 256),     # channels in each stage.
        'num_heads': (2, 2, 4, 8),          # number of head in each stage.
        
        'window_size': 7,                   # window size in block attention.
        'grid_size': 7,                     # grid size in grid attention.
        'ln_eps': 1e-8,
        'mlp_expansion': 4,
        'rd_rate': 0.25,                    # reduce rate of in_channels in SE block.
        'mlp_dropout': 0.5,
        'attn_drop': 0.5,                   # dropout rate after softmax in attention.
        'drop': 0.5,                        # dropout rate after wo in attention.
        'drop_path_rate': 0.2,              # droppath rate in encoder block.
        
        'reg_hidden_dim': 1152,             # the hidden dimension of linear in regression module.
        
        'gru_hidden_dim': 512,              # the hidden dimension of GRU.
        'gru_layer_dim': 5,                 # the layer dimension of GRU.
        
        # resource setting
        'vp_path': '/media/fang/Elements/datasets/CVIQ/viewports_8',
        'train_info_csv_path': '/home/fang/tzw1/databases/CVIQ_vp1_train_info.csv',
        'test_info_csv_path': '/home/fang/tzw1/databases/CVIQ_vp1_test_info.csv',
        'save_ckpt_path': '/home/fang/tzw1/ckpt',
        'load_ckpt_path': '',
        'tensorboard_path': '/media/h428ti/SSD/tanziwen/runs',
            
        # train setting
        'seed': 19990216,
        'model_name': 'Max360IQ-CVIQ-vp1-nogru-p1q2-epoch150',
        'dataset_name': 'CVIQ',
        'epochs': 150,
        'batch_size': 16,
        'num_workers': 8,
        'lr': 1e-4,
        'lrf': 0.01,
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'p': 1,
        'q': 2,
        'use_gru': False,
        'continue_training': False,
        'use_tqdm': True,
        'use_tensorboard': False,
        'batch_print': False,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    })
    
    if dataset == 'OIQA':
        config['num_vps'] = 2
        config['num_sequence'] = 4
        config['vp_path'] = '/media/fang/Elements/datasets/OIQA/viewports_8'
        config['train_info_csv_path'] = '/home/fang/tzw1/databases/OIQA_vp2_train_info.csv'
        config['test_info_csv_path'] = '/home/fang/tzw1/databases/OIQA_vp2_test_info.csv'
        config['save_ckpt_path'] = '/home/fang/tzw1/ckpt'
        config['load_ckpt_path'] = ''
        config['model_name'] = 'Max360IQ-OIQA-vp2-nogru-p1q2-epoch150'
        config['dataset_name'] = 'OIQA'

    elif dataset == 'JUFE':
        config['num_vps'] = 7
        config['num_sequence'] = 15
        config['use_gru'] = False
        config['epochs'] = 10
        config['mlp_dropout'] = 0.1
        config['attn_drop'] = 0.1
        config['drop'] = 0.1
        config['drop_path_rate'] = 0.1
        config['vp_path'] = '/media/fang/Data/tzw/data'
        config['train_info_csv_path'] = '/home/fang/tzw1/databases/JUFE_vp7_train_info.csv'
        config['test_info_csv_path'] = '/home/fang/tzw1/databases/JUFE_vp7_test_info.csv'
        config['save_ckpt_path'] = '/home/fang/tzw1/ckpt'
        config['load_ckpt_path'] = ''
        config['model_name'] = 'Max360IQ-JUFE-vp7-gru-p1q2-epoch10'
        config['dataset_name'] = 'JUFE'
        
    return config
