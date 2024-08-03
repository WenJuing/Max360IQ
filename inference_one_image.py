import torch
import argparse
from max360iq import Max360IQ as create_model
from config import max360iq_config
from torchvision import transforms as transforms
import os
from PIL import Image
import numpy as np

def main(cfg, args):
    """
    # The prediction range when loading different weights
    # JUFE: [1.4099148511886597, 4.076176452636719]
    # OIQA: [-0.3777390420436859, 0.101930821314454]
    # CVIQ: [-0.1471901461482048, -0.0216653586830943]
    """
    print("*****begin test*******************************************************")
    model = create_model(cfg).cuda(args.device)
    checkpoint = torch.load(args.load_ckpt_path, map_location=args.device)
    print(model.load_state_dict(checkpoint['model_state_dict'], strict=False))
    print("weights had been load!\n")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
    ])

    vs_scores = []
    for vs in sorted(os.listdir(args.test_img_path)):
        vp_tensors = []
        vp_list = sorted(os.listdir(os.path.join(args.test_img_path, vs)))
        for vp_name in sorted(vp_list):
            vp_path = os.path.join(args.test_img_path, vs, vp_name)
            vp = Image.open(vp_path)
            vp_tensor = test_transform(vp).float().unsqueeze(0)
            vp_tensors.append(vp_tensor)

        vs = torch.cat(vp_tensors)
        vs = vs.unsqueeze(0)
        # test
        model.eval()
        vs_score = model(vs.to(args.device)).item()
        vs_scores.append(vs_score)
    pred = np.mean(vs_scores)
    print("predict results:", pred)

if __name__ == '__main__':
    cfg = max360iq_config()
    cfg['use_gru'] = False  # it is recommended to load JUFE weights and enable GRUs
                            # when there is a temporal relationship between viewports.

    parse = argparse.ArgumentParser()

    parse.add_argument('--load_ckpt_path', type=str, default='/home/fang/tzw1/ckpt/paper/OIQA.pth')
    parse.add_argument('--test_img_path', type=str, default='/home/fang/tzw1/ckpt/paper/test_images/img1')
    parse.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    args = parse.parse_args()

    main(cfg, args)
