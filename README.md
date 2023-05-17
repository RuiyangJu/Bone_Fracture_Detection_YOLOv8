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
Hospital emergency departments frequently receive lots of bone fracture cases, with pediatric wrist trauma fracture accounting for the majority of them. Before pediatric surgeons perform surgery, they need to ask patients how the fracture occurred and analyze the fracture situation by interpreting X-ray images. The interpretation of X-ray images often requires a combination of techniques from radiologists and surgeons, which requires time-consuming specialized training. With the rise of deep learning in the field of computer vision, network models applying for fracture detection has become an important research topic. In this paper, we train YOLOv8 (the latest version of You Only Look Once) model on the GRAZPEDWRI-DX dataset, and use data augmentation to improve the model performance. The experimental results show that our model have reached the state-of-the-art (SOTA) real-time model performance. Specifically, compared to YOLOv8s models, the mean average precision (mAP 50) of our models improve from 0.604 and 0.625 to 0.612 and 0.631 at the input image size of 640 and 1024, respectively. To enable surgeons to use our model for fracture detection on pediatric wrist trauma X-ray images, we have designed the application "Fracture Detection Using YOLOv8 App" to assist surgeons in diagnosing fractures, reducing the probability of error analysis, and providing more useful information for surgery.

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
| Model | Size (pixels) | mAP 50 | mAP 50-95 | CPU Speed (ms) | GPU Speed (ms) | Params (M) | FLOPs (B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| YOLOv8n | 640 | 0.601 | 0.374 | 67.4 | 2.9 | 3.0 | 8.1 |
| YOLOv8s | 640 | 0.604 | 0.383 | 191.5 | 4.3 | 11.1 | 28.5 |
| YOLOv8m | 1024 | 0.631 | 0.403 | 536.4 | 5.5 | 25.8 | 78.7 |
| YOLOv8l | 1024 | 0.620 | 0.403 | 1006.3 | 7.4 | 43.6 | 164.9 |
| YOLOv8n | 640 | 0.605 | 0.387 | 212.1 | 3.3 | 3.0 | 8.1 |
| YOLOv8s | 640 | 0.622 | 0.399 | 519.5 | 4.9 | 11.1 | 28.5 |
| YOLOv8m | 1024 | 0.614 | 0.399 | 1521.5 | 10.0 | 25.8 | 78.7 |
| YOLOv8l | 1024 | 0.636 | 0.404 | 2671.1 | 15.1 | 43.6 | 164.9 | 

## Application
For research project agreement, we don't release training code, please refer to [YOLOv7 Bone Fracture Detection](https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection) and our paper for details.
