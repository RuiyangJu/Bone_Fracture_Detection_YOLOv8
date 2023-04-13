## [Fracture Detection in Pediatric Wrist Trauma X-ray Images Using YOLOv8 Algorithm](https://arxiv.org/abs/2304.05071)
### YOLOv8 architecture
<p align="center">
  <img src="img/figure_details.jpg" width="640" title="Stage-1">
</p>

### Fracture Detection Using YOLOv8 App
<p align="center">
  <img src="img/figure_application.jpg" width="640" title="Stage-2">
</p>

## Abstract
Hospital emergency departments frequently receive lots of bone fracture cases, with pediatric wrist trauma fracture accounting for the majority of them. Before pediatric surgeons perform surgery, they need to ask patients how the fracture occurred and analyze the fracture situation by interpreting X-ray images. The interpretation of X-ray images often requires a combination of techniques from radiologists and surgeons, which requires time-consuming specialized training. With the rise of deep learning in the field of computer vision, network models applying for fracture detection has become an important research topic. In this paper, YOLOv8 algorithm is used to train models on the GRAZPEDWRI-DX dataset, which includes X-ray images from 6,091 pediatric patients with wrist trauma. The experimental results show that YOLOv8 algorithm models have different advantages for different model sizes, with YOLOv8l model achieving the highest mean average precision (mAP 50) of 63.6%, and YOLOv8n model achieving the inference time of 67.4ms per X-ray image on one single CPU with low computing power. In this way, we create "Fracture Detection Using YOLOv8 App" to assist surgeons in interpreting X-ray images without the help of radiologists.

## Citation
If you find our paper useful in your research, please consider citing:

    @article{ju2023fracture,
      title={Fracture Detection in Pediatric Wrist Trauma X-ray Images Using YOLOv8 Algorithm},
      author={Ju, Rui-Yang and Cai, Weiming},
      journal={arXiv preprint arXiv:2304.05071},
      year={2023}
    }
    
## Requirements
* Linux (Ubuntu)
* Python = 3.9
* Pytorch = 1.13.1
* NVIDIA GPU + CUDA CuDNN

## Dataset
* GRAZPEDWRI-DX Dataset [(Download Link)](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193)
* Download dataset and put images and annotatation into `/GRAZPEDWRI-DX_dataset/data/images`, `/GRAZPEDWRI-DX_dataset/data/labels`.
  ```
    python split.py
  ```
   
## Model
You can get the open source code of YOLOv8 through [YOLOv8 official GitHub](https://github.com/ultralytics/ultralytics).
* Pip install ultralytics and [dependencies](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) and check software and hardware.
  ```
    %pip install ultralytics
  ```

## Experimental Results
<p align="center">
  <img src="img/figure_result.jpg" width="640" title="Stage-1">
</p>
Examples of the prediction of YOLOv8l model with the input image of 1024 in the GRAZPEDWRI-DX test set. (a) manually labeled images, (b) predicted images.

### YOLOv8 models on GRAZPEDWRI-DX dataset
| Option | Input | GT | FM↑ | p-FM↑ | PSNR↑ | DRD↓ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | \ | \ | 96.50 | 97.50 | 22.08 | 1.01 |
| 2 | \ | DWT(LL) | 96.52 | 97.70 | 22.15 | 0.99 |
| 3 | \ | DWT(LL)+Norm | **96.88** | **98.03** | **22.68** | **0.89** |
| 4 | DWT(LL) | \  | 95.94 | 96.91 | 21.31 | 1.19 |
| 5 | DWT(LL)+Norm | \ | 96.29 | 97.48 | 22.05 | 1.16 |
| 6 | DWT(LL) | DWT(LL) | 96.60 | 97.60 | 22.27 | 0.97 |
| 7 | DWT(LL)+Norm | DWT(LL)+Norm | 96.77 | 97.89 | 22.52 | 0.91 | 

## Application
For research project agreement, we don't release training code, please refer to [YOLOv7 Bone Fracture Detection](https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection) and our paper for details.
