# Small Object Detection
The course project of EI339 Artificial Intelligence 2019 Fall, SJTU. Our topic is small object detection, based on *SSD: Single Shot MultiBox Object Detector*(Wei Liu, et al. ECCV2016).
## Description
Recent great progress on object detection is stimulated by the deep learning pipelines, such as Faster R-CNN, Yolo and SSD. These pipelines indeed work well on large object with high resolution, clear appearance and structure from which the discriminative features can be learned. However, they usually fail to detect very small objects, as rich representations are difficult to learn from their poor-quality appearance and structure. Moreover, two stage detectors such as Faster R-CNN are more likely to perform better than one stage detectors, e.g., Yolo and SSD. In this project, you are required to improve detection performance on small objects based on the SSD pipeline.
## Requirement
+ opencv-python
+ matplotlib
+ tensorboardX
+ torchvision
+ pytorch
+ cython
+ instaboostfast

## DataSets

**Download VOC2007 trainval & test**

```shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval

```shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## Training 

By default, you can train our model  using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```
python train.py --model mssd
```

You can adjust the possibility for data augmentation in  `data/voc0712.py` as well.

## Evaluation

To evaluate our trained network, you can simply enter the following command:

```
python eval.py --model mssd --trained_model weights/aug_mssd.pth
```

## Performance

**Aug-MSSD** model:  VOC2007 Test

| mAP   | Extra Small | Small | Medium | Large | Extra Large |
| ----- | ----------- | ----- | ------ | ----- | ----------- |
| 79.62 | 23.40       | 57.14 | 87.13  | 94.77 | 92.49       |

The detailed result is shown in  `eval/aug_mssd.txt` .

