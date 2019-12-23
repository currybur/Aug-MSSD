import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
from models.backbones import vgg
import os


class FSSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, features, head, num_classes):
        super(FSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size
        self.transforms = nn.ModuleList(features[0])
        self.pyramids = nn.ModuleList(features[1])
        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = nn.BatchNorm2d(int(feature_layer[0][1][-1]/2)*len(self.transforms),affine=True)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        transformed = list()
        pyramids = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        sources.append(x)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)


        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 2 == 1:
                sources.append(x)

        assert len(self.transforms) == len(sources)

        upsize = (sources[0].size()[2], sources[0].size()[3])

        for k, v in enumerate(self.transforms):
            size = None if k == 0 else upsize
            transformed.append(v(sources[k], size))
        x = torch.cat(transformed, 1)
        x = self.L2Norm(x)
        for k, v in enumerate(self.pyramids):
            x = v(x)
            pyramids.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(pyramids, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        # self.up_size = up_size
        # self.up_sample = nn.Upsample(size=(up_size,up_size),mode='bilinear') if up_size != 0 else None

    def forward(self, x, up_size=None):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if up_size is not None:
            x = F.upsample(x, size=up_size, mode='bilinear')
            # x = self.up_sample(x)
        return x


def add_extras(base, feature_layer, mbox, num_classes, version):
    extra_layers = []
    feature_transform_layers = []
    pyramid_feature_layers = []
    loc_layers = []
    conf_layers = []
    in_channels = None
    feature_transform_channel = int(feature_layer[0][1][-1] / 2)
    for layer, depth in zip(feature_layer[0][0], feature_layer[0][1]):
        if 'lite' in version:
            if layer == 'S':
                extra_layers += [_conv_dw(in_channels, depth, stride=2, padding=1, expand_ratio=1)]
                in_channels = depth
            elif layer == '':
                extra_layers += [_conv_dw(in_channels, depth, stride=1, expand_ratio=1)]
                in_channels = depth
            else:
                in_channels = depth
        else:
            if layer == 'S':
                extra_layers += [
                    nn.Conv2d(in_channels, int(depth / 2), kernel_size=1),
                    nn.Conv2d(int(depth / 2), depth, kernel_size=3, stride=2, padding=1)]
                in_channels = depth
            elif layer == '':
                extra_layers += [
                    nn.Conv2d(in_channels, int(depth / 2), kernel_size=1),
                    nn.Conv2d(int(depth / 2), depth, kernel_size=3)]
                in_channels = depth
            else:
                in_channels = depth
        feature_transform_layers += [BasicConv(in_channels, feature_transform_channel, kernel_size=1, padding=0)]

    in_channels = len(feature_transform_layers) * feature_transform_channel
    for layer, depth, box in zip(feature_layer[1][0], feature_layer[1][1], mbox):
        if layer == 'S':
            pyramid_feature_layers += [BasicConv(in_channels, depth, kernel_size=3, stride=2, padding=1)]
            in_channels = depth
        elif layer == '':
            pad = (0, 1)[len(pyramid_feature_layers) == 0]
            pyramid_feature_layers += [BasicConv(in_channels, depth, kernel_size=3, stride=1, padding=pad)]
            in_channels = depth
        else:
            AssertionError('Undefined layer')
        loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
    return base, extra_layers, (feature_transform_layers, pyramid_feature_layers), (loc_layers, conf_layers)


feature_layer = [[[22, 34, 'S'], [512, 1024, 512]],[['', 'S', 'S', 'S', '', ''], [512, 512, 256, 256, 256, 256]]]

mbox = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location


def build_fssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, features_, head_ = add_extras(vgg(str(size), 3), feature_layer, mbox, num_classes, "fssd")

    return FSSD(phase, size, base_, extras_, features_, head_, num_classes)
