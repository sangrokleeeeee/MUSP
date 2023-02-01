from __future__ import absolute_import
from __future__ import division
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import math

import random
import numpy as np
import cv2


__all__ = ['resnet50', 'resnet50_fc512']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = conv3x3(inplanes, planes, stride, bias=False)
        self.bn1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=False)
        self.bn2 = norm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm, k, stride=1, downsample=None, relu=True):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.relu_on = relu
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # if isinstance(k, (list, tuple)):
        #     self.dep = Depression(planes * self.expansion)
        # else:
        #     self.dep = None
        
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # if self.dep != None:
        #     out = self.dep(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.relu_on:
            out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # bs, c, n
        bs, c, n = x.shape
        x = x.reshape(bs, -1).contiguous()
        # x = x.permute(0, 2, 1).reshape(-1, c).contiguous()
        y = self.fc(x)
        x = x * y
        x = x.reshape(bs, c, n).contiguous()
        return x


class CBN(nn.Module):
    def __init__(self, bn):
        super().__init__()
        self.bn = bn
    
    def forward(self, x):
        # with torch.no_grad():
        if self.training:
            self.bn.train()
            self.bn(x)
            self.bn.eval()
        # print(self.bn.num_batches_tracked)
        return self.bn(x)


class Depression(nn.Module):
    def __init__(self, in_channels):
        super(Depression, self).__init__()
        self.key = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.value = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.prototypes = nn.Conv2d(in_channels, 2, 1, bias=False)
        self.in_channels = in_channels

    def forward(self, x):
        key = self.key(x)
        # bs, 2, h, w
        key = self.prototypes(key)
        key = F.softmax(key / math.sqrt(self.in_channels), dim=1)[:, 0:1]
        value = self.value(x)
        
        return key * value


class ResNet(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, num_classes, loss, block, layers,
                 last_stride=2,
                 fc_dims=None,
                 dropout_p=None,
                 **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.norm = nn.BatchNorm2d
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        self.attn = kwargs['attn']
        self.multi = kwargs['multi']
        self.channel = kwargs['channel']
        self.vertical = kwargs['vertical']

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], (96, 96))
        self.layer2 = self._make_layer(block, 128, layers[1], (48, 48), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], (24, 24), stride=kwargs['stride3'])
        self.layer4 = self._make_layer(block, 512, layers[3], (24, 24), stride=1)

        if not self.attn:
            if self.vertical:
                self.classifier = nn.ModuleList([nn.Sequential(
                    nn.BatchNorm1d(512 * 4), nn.Linear(512 * 4, num_classes)
                )for _ in range(6)])
            else:
                self.global_avgpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten())
                self.classifier = nn.Sequential(
                    nn.BatchNorm1d(512 * 4), nn.Linear(512 * 4, num_classes)
                )
        else:
            self.num_prototypes = kwargs['num_prototypes']
            self.discard = kwargs['discard']
            self.num_dim = kwargs['num_dim']
            self.has_global = kwargs['has_global']
            self.key = nn.Conv2d(512 * 4, self.num_prototypes, 3, padding=1)
            self.value = nn.Conv2d(512 * 4, self.num_dim, 3, padding=1, groups=self.num_dim)
            if self.channel:
                self.c_ = SELayer(self.num_dim * self.num_prototypes)
            self.classifier = nn.ModuleList([
                nn.Linear(self.num_dim, num_classes, bias=False) for _ in range(self.num_prototypes + self.has_global - self.discard)])
            self.neck = nn.ModuleList([nn.BatchNorm1d(self.num_dim) for _ in range(self.num_prototypes + self.has_global - self.discard)])
            
        for n in self.neck:
            n.bias.requires_grad = False
        self._init_params()
        
    def last_computev3(self, features):
        feature = features[0]
        bs, c, h, w = feature.shape
        value_features = self.value(feature)#[v(f) for v, f in zip(self.value, features)]
        value_features = value_features.reshape(bs, self.num_dim, -1)
        key_features = self.key(feature)
        key_features = key_features.reshape(bs, self.num_prototypes, -1)

        weights = key_features.permute(0, 2, 1)
        weights = F.softmax(weights, dim=2)
        div_weights = weights.sum(dim=1, keepdim=True)

        # bs, c, n
        results = torch.bmm(value_features, weights) / h*w

        # channel calibration
        if self.channel:
            results = self.c_(results)
        # ratio
        area_ratios = div_weights.reshape(bs, -1)[..., :-self.discard]
        area_ratios = F.normalize(area_ratios, p=1, dim=-1)
        area_ratios = [t.squeeze(-1) for t in area_ratios.split(1, dim=-1)]
        
        results = results.split(1, dim=-1)
        results = [r.squeeze(-1) for r in results]
        global_features = []
        if self.has_global:
            global_features.append(value_features.mean(dim=-1))

        results, discard = results[:-self.discard], results[-self.discard:]
        ratio = 1 / (self.num_prototypes - self.discard)
        assert self.discard > 0
        return results, global_features, \
            area_ratios + [torch.ones([bs]).to(feature.device) * ratio for _ in range(len(global_features))], torch.prod(weights, dim=2).sum()

    def _make_layer(self, block, planes, blocks, k, stride=1, relu=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.norm, k, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.norm, k))
        layers[-1].relu_on = relu
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """
        Construct fully connected layer
        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x4]

    def forward(self, x):
        f = self.featuremaps(x)

        if not self.attn:
            # for vertical split
            # bs, c, 1, 5
            if self.vertical:
                global_f = f[0].mean(dim=[2, 3])

                f = F.adaptive_avg_pool2d(f[0], (1, 5))
                bs = f.shape[0]
                f = f.split(1, dim=-1)
                # [(bs,c) * 5]
                f = [i.flatten(start_dim=1) for i in f] + [global_f]

                if not self.training:
                    return [torch.cat([F.normalize(i) for i in f], dim=1)], [torch.ones([bs]).to(f[0].device)]
                return [c(i) for c, i in zip(self.classifier, f)], f
            
            # for baseline
            f = self.global_avgpool(f[0])
            if not self.training:
                return [f], [None]
            
            return self.classifier(f), f

        else:
            
            v, g, area, p = self.last_computev3(f)

            if not self.training:
                # return torch.cat([i for n, i in zip(self.neck, itertools.chain(*(v + g)))], dim=1)
                # return [i for i in v + g], area
                # return [n(i) for n, i in zip(self.neck, v)], area
                return [F.normalize(i) for i in v + g], area
                return torch.cat([F.normalize(i) for i in v + g], dim=1), area

            return [c(n(i)) for c, n, i in zip(self.classifier, self.neck, v + g)], v, g, p
            # return [c(n(i)) for c, n, i in zip(self.classifier, self.neck, v)], v
            # return self.classifier(self.neck(v)), v


def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print('Initialized model with pretrained weights from {}'.format(model_url))


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""


def resnet50(num_classes, loss={'xent'}, pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet50_fc512(num_classes, loss={'xent'}, pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
