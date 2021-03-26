# 导入必要的包
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


# 解析配置文件
def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """

    file = open(cfgfile, 'r')

    # store the lines in a list
    lines = file.read().split('\n')
    # 删除空行
    lines = [x for x in lines if len(x) > 0]
    # 去除注释
    lines = [x for x in lines if x[0] != '#']
    # 去除留白
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        # This marks the start of a new block
        if line[0] == "[":
            # If block is not empty, implies it is storing values of previous block.
            if len(block) != 0:
                # add it the blocks list
                blocks.append(block)
                # re-init the block
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

    # 在迭代列表之前，我们先定义变量net_info,来存储网络的信息
    # Captures  the information about the input and preprocessing
    net_info = blocks[0]

    module_list = nn.ModuleList()
    # 追踪上一层的卷积和数量
    # 初始化为3，因为图像有对应RGB通道的3个通道
    prev_filters = 3
    output_filters = []

    # 迭代模块的列表，并为每个模块创建PyTorch模块
    for index, x in enumerate(blocks[1:]):

        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list

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

            # 添加Batch Norm层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # check the activation
            # Linear or a Leaky Relu
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        # upsampling layer
        # Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        # route layer
        # 路由层有一个或两个值。当只有一个值时，它输出这一层通过该值索引的特征图
        # 当层级有两个值时，它将返回由这两个值索引的拼接特征图
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')

            # Start of a route
            start = int(x["layers"][0])
            # end, if there exists one
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

        # shortcut corresponds  to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detectoin_{}".format(index), detection)

        # 加入module_list
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
        把检测特征图转换成二维张量
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # torch.view等方法操作需要连续的Tensor
    '''
    某些Tensor操作（如transpose、permute、narrow、expand）与原Tensor是共享内存中的数据，
    不会改变底层数组的存储，但原来在语义上相邻、内存里也相邻的元素在执行这样的操作后，在语义上相邻，
    但在内存不相邻，即不连续了(is not contiguous)
    '''
    print(prediction.shape, bbox_attrs, num_anchors, grid_size)
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # 锚点的维度与 net 块的 height 和 width 属性一致。这些属性描述了输入图像的维度，比检测图的规模大（二者之商即是步幅）。因此，我们必须使用检测特征图的步幅分割锚点。
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the centre_X, centre_Y, and object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # 将网格偏移添加到中心坐标预测中

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # 按维数（行）拼接  ---- torch.cat
    # 按位数进行扩张  ------ torch.repeat
    # 在特定唯独添加插入1 ----- torch.unsqueeze
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # 将锚点应用到边界框维度中
    # log space  transform height and width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # sigmoid 激活函数应用到类别分数中
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # 我们要将检测图的大小调整到与输入图像大小一致
    prediction[:, :, :4] *= stride

    return prediction


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    # 去除数组中的重复数字，并进行排序之后输出
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = (inter_rect_x2 - inter_rect_x1 + 1) * (inter_rect_y2 - inter_rect_y1 + 1)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """
    函数的输入为预测结果、置信度（objectness 分数阈值）、num_classes（我们这里是 80）和 nms_conf（NMS IoU 阈值）
    """

    # 目标置信度阈值
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # 执行非极大值抑制
    # 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        # confidence threshholding
        # NMS

        # 移除了4个边界和1个置信度，并且确定最大值的类别的索引，及分数
        """
        torch.max(input, dim) 函数
        input是softmax函数输出的一个tensor
        dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
        """

        max_conf, max_conf_score = torch.max(image_pred[:, 5: 5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        # For PyTorch 0.4 compatibility
        # Since the above code with not raise exception for no detection
        # as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue

        # Get the various classes detected in the image
        # -1 index holds the class index
        img_classes = unique(image_pred_[:, -1])

        # 按照类别执行NMS
        for cls in img_classes:
            # perform NMS

            # get the detections with one particular class
            cls_msk = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_msk[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections  such that the entry with the maximum objectnesses
            # confidence is at the top
            # 1表示索引

            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            # Number of detections
            idx = image_pred_class.size(0)

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            #Repeat the batch_id for as many detections of class cls in the image
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    try:
        return output
    except:
        return 0
if __name__ == "__main__":
    blocks = parse_cfg('./cfg/yolov3.cfg')
    print(create_modules(blocks))
