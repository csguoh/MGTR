# MGTR

An official implement of the paper [MGTR: End-to-end Mutual Gaze Detection with Transformer]().

<img src="E:\open-source\assets\architecture.png" alt="architecture" style="zoom: 67%;" />

### Dependencies

- Python >= 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.7.1](https://pytorch.org/)
- [TorchVision>=0.8.2](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Opencv-python>=4.5.1


### Performance on mAP
| Model | AVA-LAEO | UCO-LAEO |
|---|---|---|
| MGTR | 66.2 | 64.8 |

###  Visualization

<img src="E:\open-source\assets\viz.gif" alt="viz" style="zoom: 67%;" />

### Quick Start

1. Clone this github repo.
   ```
   git@github.com:Gmbition/MGTR.git
   cd MGTR
   ```

2. Download Mutual Gaze Datasets from Baidu Drive(coming soon~).

   - [AVA-LAEO]()   password:
   - [UCO-LAEO]()  password:

3. Download our trained model from [here](https://drive.google.com/drive/folders/1Wu3ZEIfTiQ-Me8iknbPhEHMIiDWLeUaS?usp=sharing) and move them to `./data/mgtr_pretrained`.

4. Run testing for MGTR.

   ```
   python3 test.py --backbone=resnet50 --batch_size=1 --log_dir=./ --model_path=your_model_path
   ```

5. The visualization results (if set `save_image = True`) will be sorted in `./log`.

### Annotations

We annotate each mutual gaze instance in one image as a dict and the annoartion is stored in `./data`. There are four annotation json files for AVA-LAEO and UCO-LAEO training and testing respectively. The specific format of one mutual gaze instance annoatation is as follow:

`{"file_name": "scence_name/image.jpg", "width": width of the image, "height": height of the image, "gt_bboxes": [{"tag": 1, "box": a list containing the [x,y,w,h] of the box}, ...], "laeo": [{"person_1": the idx of person1, "person_2": the idx of person2, "interaction": whether looking at each other}]}`

### Acknowledgement

We sincerely thank the cool work by some very cool people, especially  [DETR]([facebookresearch/detr: End-to-End Object Detection with Transformers (github.com)](https://github.com/facebookresearch/detr)), [HoiTransformer]([bbepoch/HoiTransformer: This is the code for HOI Transformer (github.com)](https://github.com/bbepoch/HoiTransformer)).
