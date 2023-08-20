# Based on code taken from https://github.com/weiaicunzai/pytorch-cifar100

"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(
        self, in_channels, out_channels, stride=1, batch_norm=True
    ):
        super().__init__()

        self.batch_norm = batch_norm

        # residual function
        layer_list = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
        ]
        if self.batch_norm:
            layer_list.append(nn.BatchNorm2d(out_channels))
        layer_list += [
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        ]
        if self.batch_norm:
            layer_list.append(nn.BatchNorm2d(out_channels * BasicBlock.expansion))
        self.residual_function = nn.Sequential(*layer_list)

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            layer_list = [
                nn.Conv2d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            ]
            if self.batch_norm:
                layer_list.append(nn.BatchNorm2d(out_channels * BasicBlock.expansion))
            self.shortcut = nn.Sequential(*layer_list)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(
        self, block, num_block, base_width, num_classes=200, batch_norm=True,
    ):
        super().__init__()

        self.in_channels = base_width

        self.batch_norm = batch_norm

        if self.batch_norm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, base_width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, base_width, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, base_width, num_block[0], 1)
        self.conv3_x = self._make_layer(block, base_width*2, num_block[1], 2)
        self.conv4_x = self._make_layer(block, base_width*4, num_block[2], 2)
        self.conv5_x = self._make_layer(block, base_width*8, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for stride in strides:
            layer_list.append(
                block(
                    self.in_channels, out_channels, stride, self.batch_norm
                )
            )
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layer_list)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def _resnet(arch, block, num_block, base_width, num_classes, pretrained, batch_norm, model_dir="pretrained_models"):
    model = ResNet(block, num_block, base_width, num_classes, batch_norm)
    if pretrained:
        pretrained_path = "{}/{}-cifar{}.pt".format(model_dir, arch, num_classes)
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = pretrained_dict["model_state_dict"] # necessary because of our ckpt format
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet_small(input_shape, num_classes, pretrained=False, model_dir="pretrained_models"):
    return _resnet(
        "resnet-small",
        BasicBlock,
        [2, 2, 2, 2],
        8,
        num_classes,
        pretrained,
        batch_norm=True,
        model_dir=model_dir,
    )


def resnet_big(input_shape, num_classes, pretrained=False, model_dir="pretrained_models"):
    return _resnet(
        "resnet-big",
        BasicBlock,
        [2, 2, 2, 2],
        16,
        num_classes,
        pretrained,
        batch_norm=True,
        model_dir=model_dir,
    )
