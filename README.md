# Max360IQ
Pytorch implementation of "Max360IQ: Blind Omnidirectional Image Quality Assessment with Multi-axis Attention"

## :seedling:Usage

### Inference one image
* `use_gru`(True/False): it is recommended to set True when there is a temporal relationship in the viewport sequence and loading the weights trained on JUFE
* Modify the `load_ckpt_path` to load pre-trained weights
* Modify the `test_img_path` to inference one image
```python
python inference_one_image.py
```

### Train and Test
* Prepare the viewport for images by the tool `getImageViewport`
* The pre-trained weights can be downloaded at the <a href="https://drive.google.com/drive/folders/18vCXea59S9JMYSaXBAe82mxa-_6i7FFJ" target="_blank">Google drive</a>
* Run the file `train.py` and `test.py` for training and testing

## :dart:Moel Architecture

<img src="https://github.com/WenJuing/Max360IQ/blob/main/model_architecture.png">
The architecture of our proposed Max360IQ. It mainly consists of three parts: a backbone, a multi-scale feature integration (MSFI) module, and a quality regression (QR) module. Note that the GRUs component in Max360IQ is optional for optimal performance in different scenarios, i.e., non-uniformly and uniformly distorted omnidirectional images
