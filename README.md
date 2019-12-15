# Small Object Detection
The course project of EI339 Artificial Intelligence 2019 Fall, SJTU. Our topic is small object detection, based on *SSD: Single Shot MultiBox Object Detector*(Wei Liu, et al. ECCV2016).
## Description
Recent great progress on object detection is stimulated by the deep learning pipelines, such as Faster R-CNN, Yolo and SSD. These pipelines indeed work well on large object with high resolution, clear appearance and structure from which the discriminative features can be learned. However, they usually fail to detect very small objects, as rich representations are difficult to learn from their poor-quality appearance and structure. Moreover, two stage detectors such as Faster R-CNN are more likely to perform better than one stage detectors, e.g., Yolo and SSD. In this project, you are required to improve detection performance on small objects based on the SSD pipeline.
## Requirement
### Baseline
Re-implement the SSD and evaluate recall, precision, mAP and FPS on 
Pascal VOC 2007 Dataset for each kind of size category.
Note that each object is assigned to a size category, depending on the object’s percentile size within its category: extra-small (XS: bottom 10%); small (S: next 20%); medium (M: next 40%); large (L: next 20%); extra-large (XL: next 10%).
### Good work
Read some related papers and apply some novel methods to improve the performance of small object detection. Here, we regard extra-small (XS) and small (S) size category as small objects.
### Excellent work
Do you have any novel idea to improve it? Have a try.
## References
- [A pytorch implementation of SSD](https://github.com/amdegroot/ssd.pytorch)
- [A chinese blog about SSD](https://blog.csdn.net/weixin_43384257/article/details/93501343)
- [VOC2007 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)
- [What is mAP](https://www.zhihu.com/question/53405779/answer/419532990)
- [Papers on small object detection](https://github.com/tjtum-chenlab/SmallObjectDetectionList)

