# An ensemble learning approach with attention mechanism for detecting pavement distress and disaster-induced road damage
## Requirements
```
pip install -r requirements.txt
```
## Prepare datasets
1. Download **train.tar.gz** from [RDD2020](https://data.mendeley.com/datasets/5ty2wb6gvg/2). The path of the RDD2020 is like
```
datasets
└─RDD2020
    ├─train
    │  └─Czech
    │      ├─annotations
    │      └─images
    │  ├─India
    │      ├─annotations
    │      └─images
    │  ├─Japan
    │      ├─annotations
    │      └─images
```

2. Download and unzip our **SODR dataset** from [baiduyun](https://pan.baidu.com/s/1BhvnxnlPwPdHBokhK1EGkA?pwd=94y3) (94y3) or [google drive](https://drive.google.com/file/d/1uDj-ior96CTLAMNTb0XKo9j-YUaGQH0c/view?usp=sharing). Please move it to ```datasets/SODRv1```.
```
datasets
└─SODRv1
    ├─blockage
    │  └─images
    │  ├─xmls
    ├─collapse
    │  └─images
    │  ├─xmls
```

3. Run ```prepare/prepare_data.py``` to create ```datasets/train.txt```, ```datasets/val.txt```, and ```datasets/test.txt```.
```
python prepare/prepare_data.py
```

## Train
1. Please download [pre-trained weights of YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x.pt) on COCO dataset. And move it to ```weights/yolov5x.pt```.

2. Train YOLOv5:
```
python train.py --img 640 --data data/RDD2020_SODRv1.yaml --epochs 100 --cfg models/yolov5x.yaml --weights weights/yolov5x.pt --batch-size 16 --workers 0
```

3. Train YOLOv5_SE:
```
python train_yolov5_SE.py --img 640 --data data/RDD2020_SODRv1.yaml --epochs 100 --cfg models/yolov5x_SE.yaml --weights weights/yolov5x.pt --batch-size 16 --workers 0
```

4. Train YOLOv5_CA:
```
python train_yolov5_CA.py --img 640 --data data/RDD2020_SODRv1.yaml --epochs 100 --cfg models/yolov5x_CA.yaml --weights weights/yolov5x.pt --batch-size 16 --workers 0
```

**Note:** the trained weights will be saved in ```runs/train/exp/weights```, and use the *last.pt* rather than *best.pt*.

## Test
1. You can download our trained models on [baiduyun](https://pan.baidu.com/s/1NTEtPBrGtw7Ptlad8doZzw?pwd=ffvt) (ffvt) or [google drive](https://drive.google.com/drive/folders/1N3cszrA8i6FY196oCO8S4qVhn4V8filD?usp=sharing). Please move it to ```weights/train```.

2. Test YOLOv5 without test time augmentation:
```
python val.py --data data/RDD2020_SODRv1.yaml --weights weights/train/yolov5_sodr.pt --batch-size 1 --img-size 640 --conf-thres 0.2 --iou-thres 0.5 --task test
```

3. Test YOLOv5 with test time augmentation:
```
python val.py --data data/RDD2020_SODRv1.yaml --weights weights/train/yolov5_sodr.pt --batch-size 1 --img-size 640 --conf-thres 0.2 --iou-thres 0.5 --task test --augment
```

4. Test the ensemble model with test time augmentation:
```
python val.py --data data/RDD2020_SODRv1.yaml --weights weights/train/yolov5_sodr.pt weights/train/yolov5se_sodr.pt weights/train/yolov5ca_sodr.pt  --batch-size 1 --img-size 640 --conf-thres 0.2 --iou-thres 0.5 --task test --augment
```

## Acknowledgements
This code is based on [YOLOv5](https://github.com/ultralytics/yolov5/tree/v6.0). We thank the authors for sharing codes.

## Contact
If you have any questions, please contact me with email (sxwang.w@gmail.com)
