# MGTR

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2209.10930.pdf) [![Demo](https://img.shields.io/badge/Demo-Video-blue)](https://youtu.be/pLV3MNJ0M7k)

[ACCV 2022] An official implement of the paper [MGTR: End-to-end Mutual Gaze Detection with Transformer](https://arxiv.org/pdf/2209.10930.pdf).

<img  src="https://github.com/Gmbition/MGTR/blob/main/assets/image.png" width="850px">

## 📑 Dependencies

- Python >= 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.7.1](https://pytorch.org/)
- [TorchVision>=0.8.2](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Opencv-python>=4.5.1


## 💕 Performance on mAP
| Model | AVA-LAEO | UCO-LAEO |
|---|---|---|
| MGTR | 66.2 | 64.8 |

## 👀 Visualization


<img  src="https://github.com/Gmbition/MGTR/blob/main/assets/viz.gif" width="450px">

## 😀 Quick Start

1. Clone this github repo.
   ```
   git@github.com:Gmbition/MGTR.git
   cd MGTR
   ```

2. Download Mutual Gaze Datasets from Baidu Drive and put the annotation json files to `./data`. 

   - [AVA-LAEO](https://pan.baidu.com/s/1Kt02uAEr3us7iu43DYAK7g?pwd=ava1)[5.18G]      password: ava1
   - [UCO-LAEO](https://pan.baidu.com/s/1_LbgTUGzHyvwSfBVzfvNnQ?pwd=uco1)[3.84G]      password: uco1

3. Download our trained model from [here](https://drive.google.com/drive/folders/1Wu3ZEIfTiQ-Me8iknbPhEHMIiDWLeUaS?usp=sharing) and move them to `./data/mgtr_pretrained`(need to creat this new `mgtr_pretrained` file).

4. Run testing for MGTR.

   ```
   python3 test.py --backbone=resnet50 --batch_size=1 --log_dir=./ --model_path=your_model_path
   ```

5. The visualization results (if set `save_image = True`) will be sorted in `./log`.

## 📖 Annotations

We annotate each mutual gaze instance in one image as a dict and the annoataion is stored in `./data`. There are four annotation json files for AVA-LAEO and UCO-LAEO training and testing respectively. The specific format of one mutual gaze instance annoatation is as follow:

```
{
"file_name": "scence_name/image.jpg",
"width": width of the image,
"height": height of the image, 
"gt_bboxes": [{"tag": 1, 
               "box": a list containing the [x,y,w,h] of the box},
               ...],
"laeo": [{"person_1": the idx of person1, 
          "person_2": the idx of person2, 
          "interaction": whether looking at each other}]
}
```

## 📘 Citation
Please cite us if this work is helpful to you.
```
@inproceedings{guo2022mgtr,
  title={MGTR: End-to-End Mutual Gaze Detection with Transformer},
  author={Guo, Hang and Hu, Zhengxi and Liu, Jingtai},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={1590--1605},
  year={2022}
}
```

## :blush: Acknowledgement

We sincerely thank the cool work by very cool people :sunglasses: 
[DETR](https://github.com/facebookresearch/detr), [HoiTransformer](https://github.com/bbepoch/HoiTransformer).
