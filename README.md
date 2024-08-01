# Max360IQ
Pytorch implementation of "Max360IQ: Blind Omnidirectional Image Quality Assessment with Multi-axis Attention"

## :palm_tree:Data preparation
You can download databases at [JUFE](https://github.com/LXLHXL123/JUFE-VRIQA)、OIQA and [CVIQ](https://github.com/sunwei925/CVIQDatabase)
* Extract the viewports of omnidirectional images by using the tool `getImageViewport`

## :seedling:Usage

### Inference one Image
* `use_gru`(True/False): it is recommended to set True when there is a temporal relationship in the viewport sequence and loading the weights trained on JUFE
* Modify the `load_ckpt_path` to load pre-trained weights
* Modify the `test_img_path` to prepare the image data, the directory structure of a testing image is as follows:
```plaintext
Test_image/
├── vs1/
│   ├── vp1.png
│   ├── vp2.png
│   ├──   ...
│   ├── vpK.png
├── vs2/
│   ├── vp1.png
│   ├── vp2.png
│   ├──   ...
│   ├── vpK.png
├── ...
└── vsM/
    ├── vp1.png
    ├── vp2.png
    ├──   ...
    └── vpK.png
```
* Run the following code for inference one image
```python
python inference_one_image.py
```

### Train and Test
* The pre-trained weights can be downloaded at the <a href="https://drive.google.com/drive/folders/18vCXea59S9JMYSaXBAe82mxa-_6i7FFJ" target="_blank">Google drive</a>
* Edit the `config.py` for an implement
* Run the file `train.py` and `test.py` for training and testing
* If you need train our model on other databases, loading weights pre-trained on JUFE could has better training results

## :dart:Moel Architecture

<img src="https://github.com/WenJuing/Max360IQ/blob/main/model_architecture.png">
The architecture of our proposed Max360IQ. It mainly consists of three parts: a backbone, a multi-scale feature integration (MSFI) module, and a quality regression (QR) module. Note that the GRUs component in Max360IQ is optional for optimal performance in different scenarios, i.e., non-uniformly and uniformly distorted omnidirectional images

## Citation
```plaintext
@article{yan2024max360iq,
title={Max360IQ: Blind omnidirectional image quality assessment with multi-axis attention},
author={Yan, Jiebin and Tan, Ziwen and Fang, Yuming and Rao, jiale and Zuo, Yifan},
year={2024}
}
```
