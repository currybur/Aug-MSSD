"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import json
import time
import torch
import torch.utils.data as data
import cv2
import numpy as np
from copy import deepcopy
import random
from instaboostfast import get_new_data, InstaBoostConfig
import random
from copy import deepcopy

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")
possibility = 0.5

with open("data/VOC2007_segmentation.json","r") as f:
    seg_2007 = json.load(f)
with open("data/VOC2012_segmentation.json","r") as f:
    seg_2012 = json.load(f)
# {image_id:[{"bbox":[xmin,ymin,xmax,yamx](relative),"segmentation":coco_style},...],...}


def check(bbox1,bbox2):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2

    if not ((x11 < x22) and (y11 < y22) and (x21 < x12) and (y21 < y12)):
        return False
    intersection = (min(x12,x22)-max(x11,x21))*(min(y12,y22)-max(y11,y21))
    intersection = max(0, intersection)
    total = (x12-x11)*(y12-y11)+(x22-x21)*(y22-y21)
    union = total - intersection
    if float(intersection)/float(union)>0.95:
        return True
    else:
        return False


def find_seg(id_tuple,bbox):
    year = id_tuple[0][-4:]
    image_id = str(int(id_tuple[1]))
    if year=="2007":
        corr = seg_2007[image_id]
    else:
        corr = seg_2012[image_id]

    for item in corr:
        if check(item["bbox"],bbox):
            return item["segmentation"]
    raise AttributeError


def boost(img,target,img_id):
    t0 = time.time()

    anns = []
    height, width, channels = img.shape
    ori_target = deepcopy(target)
    try:
        for id in range(len(target)):
            item = target[id]
            ann= {}
            ann['bbox'] = [item[0]*width, item[1]*height, (item[2]-item[0])*width, (item[3]-item[1])*height]
            ann['segmentation'] = find_seg(img_id,item[:4])
            ann['category_id'] = id
            anns.append(ann)
        # print("t1",time.time()-t0)
        new_anns, new_img = get_new_data(anns, img, config=InstaBoostConfig(heatmap_flag=True))
        # print("t2",time.time()-t0)
        for id in range(len(target)):
            for ann in new_anns:
                if id == ann['category_id']:
                    target[id][0] = ann['bbox'][0]/float(width)
                    target[id][1] = ann['bbox'][1]/float(height)
                    target[id][2] = (ann['bbox'][0]+ann['bbox'][2])/float(width)
                    target[id][3] = (ann['bbox'][1]+ann['bbox'][3])/float(height)
        # print("t3",time.time()-t0)
    except:
        target = ori_target
        new_img = img
    # r = random.randint(1,10000)
    # cv2.imwrite("image/%d_ori.png"%r,img)
    # cv2.imwrite("image/%d_new.png"%r,new_img)

    return new_img,target

class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = float(cur_pt) / width if i % 2 == 0 else float(cur_pt) / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        # img_id = self.ids[index]
        #
        # target = ET.parse(self._annopath % img_id).getroot()
        # img = cv2.imread(self._imgpath % img_id)
        # height, width, channels = img.shape
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target, width, height)
        #
        # if self.image_set[0][1] in ("train","trainval") and random.random()<possibility:
        #     img, target = boost(img,target,img_id)
        #
        # if self.transform is not None:
        #     target = np.array(target)
        #     img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
        #     # to rgb
        #     img = img[:, :, (2, 1, 0)]
        #     # img = img.transpose(2, 0, 1)
        #     target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return im,gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.image_set[0][1] in ("train","trainval") and random.random()<possibility:
            img, target = boost(img,target,img_id)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
