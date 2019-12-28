import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint
from mmdet.core import coco_eval, results2json, wrap_fp16_model, get_classes, tensor2imgs
import cv2
import mmcv
import os

## PS: 这是用来对自己的代码自定义实现的 grad-cam 可视化, 对一般的模型都不适用...

"""
一个值得思考的问题: 
    1. 看到的针对detection进行可视化的一般都是针对于其中的 attention module 进行的可视化...
    2. 如果我们想要对 faster-rcnn 这类进行可视化的话... 选择使用最后的分类的类别分数进行可视化, 那么得到的heatmap肯定是关于
       那个 roi 的... 也就是在 feature map 上仅仅占据很小的一个范围...
    3. 所以这么说... 我的可视化还有啥用处... 也行吧... 针对单个的 roi 进行特征的可视化
"""

class FeatureExtractor(object):
    """
    用来提取网络的各种输出的参数同时获取到FPN层的梯度信息
    PS: 各种输出应该包含:
        1. 各个stage的FPN的输出: 用来生成特征提取图
        2. 各个stage的bbox的信息
        3. 各个stage的cls score的信息
        4.
    """
    def __init__(self, model, targetOutput):
        """
        :param model: CNN network, 用来进行前向与hook的添加
        :param targetOutput: 选择的模式... 其实无所谓了... 全都返回在GradCam中自己选择用啥就好了
        """
        self.model = model
        self.targetOutput = targetOutput
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, input):
        # 开始进行前向传播了
        self.gradients = []
        # print(input)
        # 获取到所有想要的信息
        # inputList, sourceList, edResultList, cls_score, bbox_pred, \
        # loss_index, loss_index_original, scores, bboxes, featureMaps, \
        # img, img_meta, bbox_results, bbox_results_original = self.model(**input)
        scores, bboxes, featureMaps, bbox_feats = self.model(**input)
        # 要为model的FPN的输出, !!!注意是输出!!! 加上hook
        for layer in featureMaps:
            layer.register_hook(self.save_gradient)
        # 暂时只需要 featureMaps, bboxes, scores
        return featureMaps, bboxes, scores

class ModelOutputs(object):
    def __init__(self, model, targetOutput):
        self.model = model
        self.targetOutput = targetOutput
        self.feature_extractor = FeatureExtractor(self.model, targetOutput)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def clean_gradients(self):
        self.feature_extractor.gradients = []

    def __call__(self, input):
        # featureMaps, bboxes, scores = self.feature_extractor(input)
        # 返回的就是 featureMaps, bboxes, scores
        return self.feature_extractor(input)



# em... 因为结构的特殊性... 所以我选择只对 FPN 的输出进行可视化... 所以target暂时不进行设置... 默认就是为 FPN 添加 hook
# 去捕捉梯度信息
class GradCam(object):
    def __init__(self, model, use_cuda, targetOutput):
        """
        :param model: 用来可视化的模型
        :param targetOutput: 期望 extractor 提取的是什么类型的信息, 比如最常见的就是 classification 的分数信息
        :param use_cuda: 是否使用 cuda
        """
        self.model = model
        # 设置模型结构为 evaluation
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.targetOutput = targetOutput
        # 这里封装了一下 ModelOutputs 用来获取 (1). 进行可视化的featuremap 以及 (2). 模型的输出
        self.extractor = ModelOutputs(self.model, targetOutput)

    # 目前看来这个东西是没啥用处的... 而且并不适合我想要的场合... 先注释掉...
    # def forward(self, input):
    #     return self.model(input)

    # selectProposalIndex 是想要选择的proposal的index... 用来多次调用的时候避免随机化的出现
    def __call__(self, input, index = None, selectProposalIndex=None):
        assert selectProposalIndex == None or len(selectProposalIndex) >= 4
        img_meta = input["img_meta"][0].data[0][0]
        if self.cuda:
            input["img"][0] = input["img"][0].cuda()
        else:
            input["img"][0] = input["img"][0].cpu()
        """
        从 extractor 获取到的信息
        featuresMaps: tuple() -> length:4 分别是 stage1 - stage4的 feature Map
                    (Shape最后两个维度不要在意, 只是大致上形容了一个比例)
                    Shape:torch.Size([1, 256, 200, 144])|torch.Size([1, 256, 100, 72])|
                          torch.Size([1, 256, 50, 36])|torch.Size([1, 256, 25, 18])|
        bboxes: torch.Size([4, 1000, 84]) -> torch.Size([4, 1000, 21, 4])
        scores: torch.Size([4, 1000, 21])
        """
        if self.targetOutput == "classification":
            featureMaps, bboxes, scores = self.extractor(input)
            maxValue, maxIndex = torch.max(scores, 2)
            camList = []
            bboxes = bboxes.view(bboxes.size(0), bboxes.size(1), -1, 4)


            # 我们会选择出来2个背景和2个前景进行特征的获取
            if selectProposalIndex:
                paintIndex = selectProposalIndex[:4]
            else:
                shuffleIndex = [i for i in range(bboxes.size(1))]
                np.random.shuffle(shuffleIndex)
                paintIndex = []
                fgCount = 0
                bgCount = 0
                # print(maxIndex.shape)
                for sindex in shuffleIndex:
                    if bgCount < 2 and torch.sum(maxIndex[:, sindex]) == 0:
                        bgCount += 1
                        paintIndex.append(sindex)
                    elif fgCount < 2 and torch.sum(maxIndex[:, sindex]) != 0:
                        fgCount += 1
                        paintIndex.append(fgCount)
                    elif fgCount+bgCount >= 4:
                        break
                assert len(paintIndex) <= 4
            print(paintIndex, "fgCount: {}  bgCount: {}".format(fgCount, bgCount))
            currentIndex = -1
            # 对每个选择出来的 proposal 进行处理
            for instanceIndex in paintIndex:
                print("deal with {}".format(instanceIndex))
                currentIndex += 1
                # 清空梯度信息
                self.extractor.clean_gradients()

                camList.append([])
                # 要一个阶段一个阶段的处理
                for stageIndex in range(4):
                    # 清空该阶段feature map的梯度信息
                    featureMaps[stageIndex].grad = None

                    if index == None:
                        index_ = maxIndex[stageIndex, instanceIndex]
                    else:
                        index_ = index
                    bbox = bboxes[stageIndex, instanceIndex, index_]
                    score = scores[stageIndex, instanceIndex, index_]

                    # 这里 one_hot 用了什么 inplace 操作... 不能进行反向传播
                    # one_hot = torch.zeros((1, scores.shape[-1]), dtype=torch.float32).requires_grad_(True)
                    # one_hot[0][index] = 1
                    # if self.cuda:
                    #     one_hot = torch.sum(one_hot.cuda() * scores[stageIndex][instanceIndex])
                    # else:
                    #     one_hot = torch.sum(one_hot * scores[stageIndex][instanceIndex])
                    # 清空模型所有阶段的梯度信息
                    for name, module in self.model.module._modules.items():
                        module.zero_grad()
                    # 进行反向传播, 这个时候 extractor 中的梯度信息数组就已经记录信息了, 下次的时候要将这个数组清空之后再进行反传
                    score.backward(retain_graph=True)
                    # print(one_hot.grad_fn)
                    # one_hot.backward(retain_graph=True)
                    # 获取grads_val
                    grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
                    # 获取目标 feature map
                    target = featureMaps[stageIndex]
                    target = target.cpu().data.numpy()
                    # 进行平均池化处理
                    weights = np.mean(grads_val, axis=(2, 3))[0, :]

                    cam = np.zeros(target.shape[2:], dtype=np.float32)

                    for i, w in enumerate(weights):
                        cam += w * target[0, i, :, :]

                    cam = np.maximum(cam, 0)
                    # print(instanceIndex, stageIndex, cam, "\n\n\n")
                    cam = cv2.resize(cam, (img_meta["pad_shape"][1], img_meta["pad_shape"][0]))  # 调整图片尺寸 PS: 这里也是要修改的, 因为实际上要调整到目标尺寸大小的
                    # 进行归一化处理(为了可视化的时候是 0-255(也就是 float 上的 0-1)的)
                    cam = cam - np.min(cam)
                    cam = cam / (np.max(cam)+1e-16)
                    # 添加camDict到camList中
                    camDict = {
                        "proposal_index": instanceIndex,
                        "cam": cam,
                        "bbox": bbox,
                        "category": index_
                    }
                    camList[currentIndex].append(camDict)

            return camList

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
    parser.add_argument("--image-path", type=str, default="/home/suilin/codes/mmdetection/data/VOCdevkit/VOC2007/JPEGImages/000001.jpg")
    parser.add_argument("--category", type=int, default=None)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def loadCfg(cfgPath):
    return Config.fromfile(cfgPath)

# 使用 mmdetection 进行model、dataset、dataloader的初始化
def modelInit(configPath, checkPointPath):
    # 进行cfg的一些配置
    cfg = loadCfg(configPath)
    # 我选择单卡测试去看结果图...
    distributed = False
    # dataset 初始化
    dataset = build_dataset(cfg.data.test)
    # dataloader 初始化
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkPointPath, map_location='cpu')
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return model, dataset, data_loader

# 进行图片的获取
def img_message_get(dataloader, index=-1):
    # index = 0
    if index < 0:
        index = np.random.randint(0, 4000)
    for i, data in enumerate(dataloader):
        if i < index:
            continue
        # img = data["img"]
        # img_meta = data["img_meta"]
        # return img, img_meta
        return data

# 存储 grad-cam 图
def savemaskList(input, maskList, CLS):
    # print(maskList)
    CLS = list(CLS)
    CLS.insert(0, "background")
    img_meta = input["img_meta"][0].data[0][0]
    # 首先进行图片的读取
    imgPath = img_meta["filename"]
    img = mmcv.imread(imgPath)
    img = mmcv.imresize(img, (img_meta["pad_shape"][1], img_meta["pad_shape"][0]))

    for proposalIndex in range(len(maskList)):
        proposalMaskList = maskList[proposalIndex]
        for stageIndex in range(len(proposalMaskList)):
            maskDict = proposalMaskList[stageIndex]
            cam = maskDict["cam"]
            bbox = maskDict["bbox"]
            proposal_index = maskDict["proposal_index"]
            category = maskDict["category"]
            # 画一个 anchor...
            x1, y1, x2, y2 = bbox
            left_top = (int(x1), int(y1))
            right_bottom = (int(x2), int(y2))
            # print(cam.shape)

            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            # cv2.imwrite("heatmap_{}_{}_{}.jpg".format(proposal_index, stageIndex, CLS[category]), heatmap)
            heatmap = np.float32(heatmap) / 255
            # print(np.sum(heatmap))
            cam = heatmap + np.float32(img/255)
            cam = cam / np.max(cam)
            cam = np.uint8(cam*255)
            cv2.rectangle(
                cam, left_top, right_bottom, color=(0, 0, 255), thickness=1)
            cv2.imwrite("cam_{}_{}_{}.jpg".format(proposal_index, stageIndex, CLS[category]), cam)
# 主题流程
if __name__ == "__main__":
    """
        使用方法: python GradCam.py [--image-path <path_to_image>] [--use-cuda]
        整体流程:
            (1). 首先使用 opencv 加载图片
            (2). 对图片进行一定的处理
            (3). 首先使用处理过的图片进行一遍正向传播, 得到特征/分数等信息
            (4). 计算 localization map (Grad-CAM, Guided Grad-CAM(暂时没有实现))
        """
    # 首先读取参数:
    args = get_args()

    # 初始化各个参数
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    configPath = "/home/suilin/codes/mmdetection/myCfgs/myDN12.py"
    checkPointPath = "/data2/suilin/myDN12/epoch_18.pth"
    target = "classification"

    # 加载想要的模型(这里用函数封装一下吧)
    model, dataset, dataloader = modelInit(configPath, checkPointPath)

    # 使用 GradCam 类进行上面模型的封装, 添加hook等信息方便获取到指定的 feature map 等
    # 信息去进行可视化
    grad_cam = GradCam(model=model, use_cuda=args.use_cuda, targetOutput=target)

    # 进行图片信息的处理:
    # 这里选择直接用 dataloader 进行图片的提取就好了... 不想太费劲了...
    data = img_message_get(dataloader)

    input = {
        "messageType":"gradCam",
		"return_loss": False,
		"see": True,
		**data
    }

    # 想要使用的类别
    target_index = args.category

    # 到此 grad_cam 已经获得了, 后面要进行的其实是 guided grad-cam 的相关信息的获取(也就是论文中提到的fine-grand的信息图)
    maskList = grad_cam(input, target_index)
    # 进行图片的保存
    savemaskList(input, maskList, model.module.CLASSES)
