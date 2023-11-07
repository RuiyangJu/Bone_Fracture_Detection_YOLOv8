# Fracture Detection in Pediatric Wrist Trauma X-ray Images Using YOLOv8 Algorithm

>[Fracture Detection in Pediatric Wrist Trauma X-ray Images Using YOLOv8 Algorithm](https://arxiv.org/abs/2304.05071)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fracture-detection-in-pediatric-wrist-trauma/object-detection-on-grazpedwri-dx)](https://paperswithcode.com/sota/object-detection-on-grazpedwri-dx?p=fracture-detection-in-pediatric-wrist-trauma)

## Abstract
Hospital emergency departments frequently receive lots of bone fracture cases, with pediatric wrist trauma fracture accounting for the majority of them. Before pediatric surgeons perform surgery, they need to ask patients how the fracture occurred and analyze the fracture situation by interpreting X-ray images. The interpretation of X-ray images often requires a combination of techniques from radiologists and surgeons, which requires time-consuming specialized training. With the rise of deep learning in the field of computer vision, network models applying for fracture detection has become an important research topic. In this paper, we train YOLOv8 (the latest version of You Only Look Once) model on the GRAZPEDWRI-DX dataset, and use data augmentation to improve the model performance. The experimental results show that our model have reached the state-of-the-art (SOTA) real-time model performance. Specifically, compared to YOLOv8s models, the mean average precision (mAP 50) of our models improve from 0.604 and 0.625 to 0.612 and 0.631 at the input image size of 640 and 1024, respectively. To enable surgeons to use our model for fracture detection on pediatric wrist trauma X-ray images, we have designed the application "Fracture Detection Using YOLOv8 App" to assist surgeons in diagnosing fractures, reducing the probability of error analysis, and providing more useful information for surgery.

### YOLOv8 architecture
<p align="center">
  <img src="img/figure_details.jpg" width="640" title="details">
</p>

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
    pip install ultralytics
  ```

## CLI
### Train
```
  yolo train model=yolov8n.pt data=meta.yaml epochs=100 batch=16 imgsz=640 save=True workers=4 pretrained=yolov8n.pt optimizer=Adam lr0=0.001
```

## Trained Model
Use gdown to download the trained model from our GitHub:
```
  gdown https://github.com/RuiyangJu/Bone_Fracture_Detection_YOLOv8/releases/download/Trained_Model/trained_model.zip
```

## Experimental Results

<p align="center">
  <img src="img/figure_640.jpg" width="640" title="640">
</p>

<p align="center">
  <img src="img/figure_1024.jpg" width="640" title="1024">
</p>

<p align="center">
  <img src="img/figure_result.jpg" width="640" title="result">
</p>
The prediction examples of our model on the pediatric wrist trauma X-ray images. (a) the manually labeled images, (b) the predicted images.

## Application
For research project agreement, we don't release APP code, please refer to [YOLOv7 Bone Fracture Detection](https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection) and our paper for details.

### Fracture Detection Using YOLOv8 App
<p align="center">
  <img src="img/figure_application.jpg" width="640" title="application">
</p>
