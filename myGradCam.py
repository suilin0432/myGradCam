import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse

"""
PS: 重新梳理一下 pytorch-grad-cam 项目实现的逻辑, 并且更新了一下代码适用性 -> 1.0 以上支持版本
原 pytorch-grad-cam 的项目地址: https://github.com/jacobgil/pytorch-grad-cam
"""

# 图片信息的处理:
def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    input = torch.from_numpy(preprocessed_img).requires_grad_(True)
    input = input.unsqueeze(0)
    return input

# 进行 Grad-Cam 的保存
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class FeatureExtractor():
    """
    用来提取 activations 并且存储梯度信息
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        # 就是简单的存储一下 梯度信息而已
        self.gradients.append(grad)

    def __call__(self, input):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            input = module(input)
            if name in self.target_layers:
                input.register_hook(self.save_gradient)
                outputs += [input]
        return outputs, input

class ModelOutputs():
    """
    用来进行模型的 forward, 获取模型的输出以及绘制 localization map 需要的特定层次的feature map
    """
    def __init__(self, model, target_layers):
        self.model = model
        # 用来提取模型中想要用来绘制 localization map 的特征层次
        # PS: 需要更改, 因为同样, 这个 model.features 只适用于 VGG 和一些 model architecture 中包含 features 模块的网络
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, input):
        # 看来这个 ModelOutputs 也就是简单的封装了一下 feature_extractor 而已... 所有信息同样是在 feature_extractor 中进行处理并返回的
        target_activations, output = self.feature_extractor(input)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


class GradCam(object):
    def __init__(self, model, target_layer_names, use_cuda):
        """
        :param model: 用来可视化的模型
        :param target_layer_names: 想要生成可视化的选择的 feature map 层的名字
        :param use_cuda: 是否使用 cuda
        """
        self.model = model
        # 设置模型结构为 evaluation
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        # 这里封装了一下 ModelOutputs 用来获取 (1). 进行可视化的featuremap 以及 (2). 模型的输出
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        # 如果没有声明类别的时候会选择使用值最大的类别进行
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        # 获取到类别的维度创建一个全零数组
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        # 其实只是为了拿到那个最大的激活值而已... 为什么要搞这么麻烦...
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # 首先要清空模型的梯度信息
        # PS: 这里使用的是是 features ... 然而这只是针对于 vgg 的... 所以后续还是要改的...
        """
        使用 
        for name, module in self.model._modules.items():
            module.zero_grad()
        应该是对的. 但是要注意的是其实 这个 for 循环拿到的 module 内部也包含了好多小模块... 如果需要细致分类的话还是需要一定处理的
        """
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        # 获取到梯度信息后面要进行平均作为这个feature map的权重
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        # 获取到目标 feature map, 并将其转化为 numpy.ndarray
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        # 对 target 进行平均处理提取到 各个 channel 的 权重
        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        # 先分配给最后的 cam 图一个空间
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224)) #调整图片尺寸 PS: 这里也是要修改的, 因为实际上要调整到目标尺寸大小的
        # 进行归一化处理(为了可视化的时候是 0-255(也就是 float 上的 0-1)的)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam




# 进行命令行输入参数的解析
def get_args():
    """
    包含参数:
        --use-cuda: 是否使用 cuda 默认为 True
        --image-path: 想要进行可视化的图片的路径
        --category: 想要查看的类别编号
    :return: args
    """
    parser = argparse.ArgumentParser()
    # action = "store_true" 表示只要包含 --use-cuda 这个参数就是为 True
    # PS: 还有其他各种的 action, 以及可以去自定义 action
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="use cuda acceleration or not")
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--category", type=int, default=None)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def modelInit():
    model = models.vgg19(pretrained=True)
    return model

# 开始进入主体流程了
if __name__ == "__main__":
    """
    使用方法: python GradCam.py [--image-path <path_to_image>] [--use-cuda]
    整体流程:
        (1). 首先使用 opencv 加载图片
        (2). 对图片进行一定的处理
        (3). 首先使用处理过的图片进行一遍正向传播, 得到特征/分数等信息
        (4). 计算各种 localization map (Grad-CAM, Guided Grad-CAM)
    """

    # 首先读取参数:
    args = get_args()

    # 加载想要的模型(这里用函数封装一下吧)
    model = modelInit()

    # 使用 GradCam 类进行上面模型的封装, 添加hook等信息方便获取到指定的 feature map 等
    # 信息去进行可视化
    grad_cam = GradCam(model=model, target_layer_names=["35"], use_cuda=args.use_cuda)

    # 进行图片信息的处理:
    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    # 想要使用的类别
    target_index = args.category

    # 到此 grad_cam 已经获得了, 后面要进行的其实是 guided grad-cam 的相关信息的获取(也就是论文中提到的fine-grand的信息图)
    mask = grad_cam(input, target_index)
    # 进行图片的保存
    show_cam_on_image(img, mask)















