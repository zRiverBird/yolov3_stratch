# 导入必要的包
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

#解析配置文件
def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """

    file = open(cfgfile, 'r')

    #store the lines in a list
    lines = file.read().split('\n')
    #删除空行
    lines = [x for x in lines if len(x) > 0]
    #去除注释
    lines = [x for x in lines if x[0] != '#']
    #去除留白
    lines = [x.rstrip().lstrip() for x in lines]


    block = {}
    blocks = []

    for line in lines:
        # This marks the start of a new block
        if line[0] == "[":
            # If block is not empty, implies it is storing values of previous block.
            if len(block) != 0:
                #add it the blocks list
                blocks.append(block)
                #re-init the block
                block = {}
                # rstrip 去除留白
                block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)

    return blocks

# 创建构建快
def create_modules(blocks):
    """
    返回一个nn.ModuleList
    """

    #在迭代列表之前，我们先定义变量net_info,来存储网络的信息
    #Captures  the information about the input and preprocessing
    net_info = blocks[0]

    module_list = nn.ModuleList()
    #追踪上一层的卷积和数量
    #初始化为3，因为图像有对应RGB通道的3个通道
    prev_filters = 3
    output_filters = []

    #迭代模块的列表，并为每个模块创建PyTorch模块
    for index, x in enumerate(blocks[1:]):
        print(x, index)
        module = nn.Sequential()

        #check the type of block
        #create a new module for the block
        #append to module_list

        # 得到层信息
        if (x['type'] == 'convolutional'):

            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # 添加卷积层
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            #添加Batch Norm层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            #check the activation
            #Linear or a Leaky Relu
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        #upsampling layer
        #Bilinear2dUpsampling
        elif (x["type"] == "unsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        #route layer
        #路由层有一个或两个值。当只有一个值时，它输出这一层通过该值索引的特征图
        #当层级有两个值时，它将返回由这两个值索引的拼接特征图
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')

            #Start of a route
            start = int(x["layers"][0])
            #end, if there exists one
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()

            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        #shortcut corresponds  to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detectoin_{}".format(index), detection)

        #加入module_list
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

if __name__ == "__main__":
    blocks = parse_cfg('./cfg/yolov3.cfg')
    print(create_modules(blocks))