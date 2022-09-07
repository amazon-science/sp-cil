# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

logger = logging.getLogger("mylogger")

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet2Br(nn.Module):
    def __init__(self, block, layers, num_classes=1000, n_freeze=0):
        self.n_classes = num_classes
        self.n_freeze = min(n_freeze, 5)
        self.inplanes = 64
        super(ResNet2Br, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_1 = self._make_layer(block, 512, layers[3], stride=2)
        self.inplanes = 256 * block.expansion
        self.layer4_2 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.n_freeze > 0:
            self._freeze_layers()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer4_1(x)
        x2 = self.layer4_2(x)

        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)

        x_f = torch.cat([x1, x2], dim=1)

        x = self.fc(x_f)

        if not feat:
            return x

        return x, x1, x2

    def _freeze_layers(self):
        for module in list(self.children())[: self.n_freeze + 4]:
            for param in module.parameters():
                param.requires_grad = False

        if self.n_freeze > 1:
            print("First {} layers of resnet are frozen.".format(self.n_freeze))
            logger.info("First {} layers of resnet are frozen.".format(self.n_freeze))
        else:
            print("First layer of resnet is frozen.".format(self.n_freeze))
            logger.info("First layer of resnet is frozen.".format(self.n_freeze))

    def train(self, mode=True):
        """Function overloaded from https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        The revision is for freezing batch norm parameters of designated layers.
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for i, module in enumerate(self.children()):
            if self.n_freeze > 0 and i < (self.n_freeze + 4):
                module.train(False)
            else:
                module.train(mode)

        return self

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)
            else:
                print(name)

    def load_state_dict2(self, state_dict1, state_dict2):
        own_state = self.state_dict()
        # load the 1st branch
        for name, param in state_dict1.items():
            name = name.replace("layer4", "layer4_1")
            if name in own_state:
                own_state[name].copy_(param)
            else:
                print(name)

        # load the 2nd branch
        for name, param in state_dict2.items():
            name = name.replace("layer4", "layer4_2")
            if name in own_state:
                own_state[name].copy_(param)
            else:
                print(name)


class ResNet2BrL3(ResNet2Br):
    def __init__(self, block, layers, num_classes=1000, n_freeze=0):
        self.n_classes = num_classes
        self.n_freeze = min(n_freeze, 6)
        self.inplanes = 64
        super(ResNet2Br, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_1 = self._make_layer(block, 256, layers[2], stride=2)
        self.inplanes = 128 * block.expansion
        self.layer3_2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_1 = self._make_layer(block, 512, layers[3], stride=2)
        self.inplanes = 256 * block.expansion
        self.layer4_2 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.n_freeze > 0:
            self._freeze_layers()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3_1(x)
        x2 = self.layer3_2(x)
        x1 = self.layer4_1(x1)
        x2 = self.layer4_2(x2)

        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)

        x_f = torch.cat([x1, x2], dim=1)

        x = self.fc(x_f)

        if not feat:
            return x

        return x, x1, x2

    def _freeze_layers(self):
        for module in list(self.children())[: self.n_freeze + 4]:
            for param in module.parameters():
                param.requires_grad = False

        if self.n_freeze > 1:
            print("First {} layers of resnet are frozen.".format(self.n_freeze))
            logger.info("First {} layers of resnet are frozen.".format(self.n_freeze))
        else:
            print("First layer of resnet is frozen.".format(self.n_freeze))
            logger.info("First layer of resnet is frozen.".format(self.n_freeze))

    def train(self, mode=True):
        """Function overloaded from https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        The revision is for freezing batch norm parameters of designated layers.
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for i, module in enumerate(self.children()):
            if self.n_freeze > 0 and i < (self.n_freeze + 4):
                module.train(False)
            else:
                module.train(mode)

        return self

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)
            else:
                print("Extra parameter not found: " + name)

    def load_state_dict2(self, state_dict1, state_dict2):
        own_state = self.state_dict()
        # load the 1st branch
        for name, param in state_dict1.items():
            name = name.replace("layer3", "layer3_1")
            name = name.replace("layer4", "layer4_1")
            if name in own_state:
                own_state[name].copy_(param)
            else:
                print("Extra parameter not found: " + name)

        # load the 2nd branch
        for name, param in state_dict2.items():
            name = name.replace("layer3", "layer3_2")
            name = name.replace("layer4", "layer4_2")
            if name in own_state:
                own_state[name].copy_(param)
            else:
                print("Extra parameter not found: " + name)


class ResNetNBr(nn.Module):
    def __init__(self, block, layers, num_classes=1000, n_freeze=0, n_branch=10):
        self.n_classes = num_classes
        self.n_freeze = min(n_freeze, 5)
        self.inplanes = 64
        super(ResNetNBr, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = nn.ModuleList()
        for idx in range(n_branch):
            self.inplanes = 256 * block.expansion
            self.layer4.append(self._make_layer(block, 512, layers[3], stride=2))
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.n_freeze > 0:
            self._freeze_layers()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_list = [layer(x) for layer in self.layer4]

        x_list = [self.avgpool(x) for x in x_list]
        x_list = [x.view(x.size(0), -1) for x in x_list]

        x_f = torch.cat(x_list, dim=1)

        x = self.fc(x_f)

        if not feat:
            return x

        return (x, *x_list)

    def _freeze_layers(self):
        for module in list(self.children())[: self.n_freeze + 4]:
            for param in module.parameters():
                param.requires_grad = False

        if self.n_freeze > 1:
            print("First {} layers of resnet are frozen.".format(self.n_freeze))
            logger.info("First {} layers of resnet are frozen.".format(self.n_freeze))
        else:
            print("First layer of resnet is frozen.".format(self.n_freeze))
            logger.info("First layer of resnet is frozen.".format(self.n_freeze))

    def train(self, mode=True):
        """Function overloaded from https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        The revision is for freezing batch norm parameters of designated layers.
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for i, module in enumerate(self.children()):
            if self.n_freeze > 0 and i < (self.n_freeze + 4):
                module.train(False)
            else:
                module.train(mode)

        return self

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)
            else:
                print(name)

    def load_state_dict2(self, *state_dicts):
        own_state = self.state_dict()
        for idx, state_dict in enumerate(state_dicts):
            # load the idx-th branch
            for name, param in state_dict.items():
                name = name.replace("layer4", "layer4.{}".format(idx))
                if name in own_state:
                    own_state[name].copy_(param)
                else:
                    print(name)


def resnet10_2br(pretrained=False, **kwargs):
    """Constructs a ResNet-10 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet2Br(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet10_2brl3(pretrained=False, **kwargs):
    """Constructs a ResNet-10 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet2BrL3(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet10_nbr(pretrained=False, **kwargs):
    """Constructs a ResNet-10 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetNBr(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18_2br(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet2Br(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet18_nbr(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetNBr(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34_2br(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet2Br(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50_2br(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet2Br(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model


def resnet101_2br(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet2Br(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152_2br(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet2Br(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model
