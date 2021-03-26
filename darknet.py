from util import *


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    # Resize to the input dimension
    img = cv2.resize(img, (416, 416))
    # img[：，：，：：-1]的作用就是实现RGB到BGR通道的转换
    # BGR -> RGB | H W C -> C H W
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    # Add a channel at 0(for batch) | Normalise
    img_ = img_[np.newaxis, :, :, :] / 255.0
    # Convert to float
    img_ = torch.from_numpy(img_).float()
    # convert to Variable
    img_ = Variable(img_)

    return img_


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def foward(self, x, CUDA):
        modules = self.blocks[1:]

        # 为了路由层缓存输出
        outputs = {}

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            # 卷基层和上采样层
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            # 路由层/shortcut层
            # 使用torch.cat 函数将两个特征图级联起来
            # 在 PyTorch 中，卷积层的输入和输出的格式为B X C X H X W
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0] > 0):
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1] > 0):
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            # yolo检测层
            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                # if no collector has been intialised
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    """
    我们必须小心地读取权重，因为权重只是以浮点形式储存，
    没有其它信息能告诉我们到底它们属于哪一层。所以如果读取错误，
    那么很可能权重加载就全错了，模型也完全不能用。因此，只阅读浮点数，
    无法区别权重属于哪一层。因此，我们必须了解权重是如何存储的。
    """

    # 加载权重
    def load_weights(self, weightfile):
        # Open the weights  file
        fp = open(weightfile, "rb")

        #  The first 5 values are header information
        # 1.Major version number
        # 2.Minor Version number
        # 3.Subversion number
        # 4, 5. Images seen by the network(during training)

        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    # torch.numel 返回的是个数
                    num_bn_biases = bn.bias.numel()

                    # Load the weight
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr += num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights  for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


if __name__ == "__main__":
    model = Darknet("cfg/yolov3.cfg")
    print(model)
    model.load_weights("./yolov3.weights")

    # inp = get_test_input()
    # pred = model.foward(inp, False)
