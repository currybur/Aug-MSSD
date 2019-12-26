"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import random
from instaboostfast import get_new_data, InstaBoostConfig
import pycocotools.mask as cocomask
from copy import deepcopy
import time


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
#VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")
VOC_ROOT = "/home/yolo/VOCdevkit/"

with open("data/VOC2007_segmentation.json","r") as f:
    seg_2007 = json.load(f)
with open("data/VOC2012_segmentation.json","r") as f:
    seg_2012 = json.load(f)
with open("/home/yolo/ssd/data/index.json","r") as f:
    SOJ_INDEX = json.load(f)
with open("/home/yolo/ssd/data/train_index_2007.json","r") as f:
   TRAIN_INDEX_07 = list(json.load(f).keys())
with open("/home/yolo/ssd/data/train_index_2012.json","r") as f:
   TRAIN_INDEX_12 = list(json.load(f).keys())

copy_possibility = 0.5
scale_possibility = 0.5


def load_index():
    """
    :return: {img_num1: [ [xmin, xmax, ymin, ymax], [xmin, xmax, ymin, ymax], ....]}
    """
    f = open('/home/yolo/ssd/data/index.json','r')
    small_object_imgs = json.load(f)
    p = open('/home/yolo/ssd/data/train_index_2007.json', 'r')
    q = open('/home/yolo/ssd/data/train_index_2012.json', 'r')
    train_imgs = json.load(p)
    tr12 = json.load(q)
    f.close()
    p.close()
    q.close()
    return train_imgs, tr12, small_object_imgs


def check(bbox1,bbox2):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2

    if not ((x11 < x22) and (y11 < y22) and (x21 < x12) and (y21 < y12)):
        return False
    intersection = (min(x12,x22)-max(x11,x21))*(min(y12,y22)-max(y11,y21))
    intersection = max(0, intersection)
    total = (x12-x11)*(y12-y11)+(x22-x21)*(y22-y21)
    union = total - intersection
    # print(float(intersection)/float(union))
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


def cocoseg_to_binary(seg, height, width):
    """
    COCO style segmentation to binary mask
    :param seg: coco-style segmentation
    :param height: image height
    :param width: image width
    :return: binary mask
    """
    if type(seg) == list:
        rle = cocomask.frPyObjects(seg, height, width)
        rle = cocomask.merge(rle)
        mask = cocomask.decode([rle])
    elif type(seg['counts']) == list:
        rle = cocomask.frPyObjects(seg, height, width)
        mask = cocomask.decode([rle])
    else:
        rle = cocomask.merge(seg)
        mask = cocomask.decode([rle])
    assert mask.shape[2] == 1
    return mask[:, :, 0]


# 按bbox来复制小物体
def bbox_copy(id, img, target, year):
    if id not in SOJ_INDEX:
        return img
    objects = SOJ_INDEX[id]  # [xmin, xmax, ymin, ymax,
                                        #  name, pose, truncated, difficult]s
    xmins = [int(float(i.text)) for i in target.iter('xmin')]
    xmaxs = [int(float(i.text)) for i in target.iter('xmax')]
    ymins = [int(float(i.text)) for i in target.iter('ymin')]
    ymaxs = [int(float(i.text)) for i in target.iter('ymax')]
    width = int(target.find('size').find('width').text)
    height = int(target.find('size').find('height').text)

    for obj in objects:
        if random.random() > copy_possibility:
            continue
        # 获得小物体的信息
        [xmin, xmax, ymin, ymax, name, pose, truncated, difficult] = obj
        xmin = int(xmin)
        xmax =int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)
        wid, hei = xmax-xmin, ymax-ymin

        # 随机生成位置来复制小物体
        count = 0
        for i in range(10):
            flag = 1
            try:
                center = (random.randrange(wid//2+1, width-wid//2),random.randrange(hei//2+1,height-hei//2))
            except:
                continue
            for j in range(len(xmins)):  # 和所有方块检查相交
                x1,x2,y1,y2 = xmins[j],xmaxs[j],ymins[j],ymaxs[j]
                c = ((x1+x2)/2, (y1+y1)/2)

                # 只要有一个相交
                if abs(c[0]-center[0])<wid/2+(x2-x1)/2\
                        and abs(c[1]-center[1])<hei/2+(y2-y1)/2:
                    flag = 0
                    break
            if flag == 0:  # 该位置不行，再找一个
                continue

            # 位置可行，开始复制
            count+=1
            item = img[ymin : ymax,xmin:xmax, : ]
            # cv2.imshow('1',item)
            # cv2.waitKey()
            img[center[1]-(hei+1)//2:center[1]+hei//2,\
            center[0]-(wid+1)//2:center[0]+wid//2, :] = item
            new_xmin = center[0]-(wid+1)//2
            new_xmax =  center[0]+wid//2
            new_ymin = center[1]-(hei+1)//2
            new_ymax = center[1]+hei//2
            xmins.append(new_xmin)
            xmaxs.append(new_xmax)
            ymins.append(new_ymin)
            ymaxs.append(new_ymax)

            # 加入annotations
            node_obj = ET.Element('object')
            node_name = ET.Element('name')
            node_name.text = name
            node_pose = ET.Element('pose')
            node_pose.text = pose
            node_tru = ET.Element('truncated')
            node_tru.text = truncated
            node_diff = ET.Element('difficult')
            node_diff.text = difficult
            node_box = ET.Element('bndbox')
            node_xi = ET.Element('xmin')
            node_xi.text = new_xmin
            node_yi = ET.Element('ymin')
            node_yi.text = new_ymin
            node_xa = ET.Element('xmax')
            node_xa.text = new_xmax
            node_ya = ET.Element('ymax')
            node_ya.text = new_ymax
            node_box.append(node_xi)
            node_box.append(node_yi)
            node_box.append(node_xa)
            node_box.append(node_ya)
            node_obj.append(node_name)
            node_obj.append(node_pose)
            node_obj.append(node_tru)
            node_obj.append(node_diff)
            node_obj.append(node_box)
            target.append(node_obj)

            if count==4:
                break


    return img, target


# 按seg来复制小物体并缩放
def seg_copy(id, img, target, year):
    if id not in SOJ_INDEX:
        return img
    objects = SOJ_INDEX[id]  # [xmin, xmax, ymin, ymax,
                                        #  name, pose, truncated, difficult]s
    xmins = [int(float(i.text)) for i in target.iter('xmin')]
    xmaxs = [int(float(i.text)) for i in target.iter('xmax')]
    ymins = [int(float(i.text)) for i in target.iter('ymin')]
    ymaxs = [int(float(i.text)) for i in target.iter('ymax')]
    # print(xmins)
    width = int(target.find('size').find('width').text)
    height = int(target.find('size').find('height').text)

    for obj in objects:

        # 获得小物体的信息
        [xmin, xmax, ymin, ymax, name, pose, truncated, difficult] = obj
        xmin = int(xmin)
        xmax =int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)
        wid, hei = xmax-xmin, ymax-ymin
        wid = min(random.randrange(int(0.8*wid), int(1.2*wid)),width-2)
        hei = min(random.randrange(int(0.8*hei), int(1.2*hei)), height-2)

        bbox = [xmin/width, ymin/height, xmax/width, ymax/height]
        # print(bbox)
        id_tuple = (year, id)
        seg = find_seg(id_tuple, bbox)
        if seg == None:
            continue
        # print(seg)
        mask = cocoseg_to_binary(seg, height, width)
        item = img.copy()
        for i in range(3):
            item[:,:,i] = item[:,:,i] * mask
        item = item[ymin:ymax, xmin:xmax, :]
        item = cv2.resize(item, (wid, hei))
        # 随机生成位置来复制小物体
        count = 0
        for i in range(10):
            flag = 1
            center = (random.randrange(wid//2+1, width-wid//2),random.randrange(hei//2+1,height-hei//2))

            for j in range(len(xmins)):  # 和所有方块检查相交
                x1,x2,y1,y2 = xmins[j],xmaxs[j],ymins[j],ymaxs[j]
                c = ((x1+x2)/2, (y1+y1)/2)

                # 只要有一个相交
                if abs(c[0]-center[0])<wid/2+(x2-x1)/2\
                        and abs(c[1]-center[1])<hei/2+(y2-y1)/2:
                    flag = 0
                    break
            if flag == 0:  # 该位置不行，再找一个
                continue

            # 位置可行，开始复制
            count+=1

            img[center[1]-(hei+1)//2:center[1]+hei//2, center[0]-(wid+1)//2:center[0]+wid//2, :][item>0] = item[item>0]
            # cv2.imshow('1',item)
            # cv2.waitKey()
            # img[center[1]-(hei+1)//2:center[1]+hei//2, center[0]-(wid+1)//2:center[0]+wid//2, :] = item
            new_xmin = center[0]-(wid+1)//2
            new_xmax =  center[0]+wid//2
            new_ymin = center[1]-(hei+1)//2
            new_ymax = center[1]+hei//2
            xmins.append(new_xmin)
            xmaxs.append(new_xmax)
            ymins.append(new_ymin)
            ymaxs.append(new_ymax)

            # 加入annotations
            node_obj = ET.Element('object')
            node_name = ET.Element('name')
            node_name.text = name
            node_pose = ET.Element('pose')
            node_pose.text = pose
            node_tru = ET.Element('truncated')
            node_tru.text = truncated
            node_diff = ET.Element('difficult')
            node_diff.text = difficult
            node_box = ET.Element('bndbox')
            node_xi = ET.Element('xmin')
            node_xi.text = new_xmin
            node_yi = ET.Element('ymin')
            node_yi.text = new_ymin
            node_xa = ET.Element('xmax')
            node_xa.text = new_xmax
            node_ya = ET.Element('ymax')
            node_ya.text = new_ymax
            node_box.append(node_xi)
            node_box.append(node_yi)
            node_box.append(node_xa)
            node_box.append(node_ya)
            node_obj.append(node_name)
            node_obj.append(node_pose)
            node_obj.append(node_tru)
            node_obj.append(node_diff)
            node_obj.append(node_box)
            target.append(node_obj)

            if count==4:
                break

    return img, target


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
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
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
                 dataset_name='VOC0712', aug=0):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.aug = aug
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        try:
            if random.random() < copy_possibility:
                if img_id[1] not in SOJ_INDEX:  # 图中没有包含小物体，则过采样
                    # if '_' in img_id[1]:
                    #     rand_id = random.choice(TRAIN_INDEX_12)
                    # else:
                    #     rand_id = random.choice(TRAIN_INDEX_07)
                    # img_id = (img_id[0], rand_id)
                    target = ET.parse(self._annopath % img_id).getroot()
                    img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
                else:  # 图中包含小物体，则复制
                    target = ET.parse(self._annopath % img_id).getroot()
                    img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
                    img, target = seg_copy(img_id[1], img, target, img_id[0][-4:])
                height, width, channels = img.shape
                if self.target_transform is not None:
                    target = self.target_transform(target, width, height)

            else:
                target = ET.parse(self._annopath % img_id).getroot()
                img = cv2.imread(self._imgpath % img_id)

                height, width, channels = img.shape

                if self.target_transform is not None:
                    target = self.target_transform(target, width, height)

                # if random.random()<scale_possibility:
                #     img, target = boost(img,target,img_id)
        except:
            target = ET.parse(self._annopath % img_id).getroot()
            img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
            height, width, channels = img.shape
            if self.target_transform is not None:
                target = self.target_transform(target, width, height)

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
