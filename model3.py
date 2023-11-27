# from torchvision.io.image import read_image
# from torchvision.models.resnet import resnet18

# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained on COCO
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torch, cv2, base64
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda")

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

#---- custom convolution layer ----
class CustomConv2d(torch.nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, dilation=1, padding=0, stride=1):
        super(CustomConv2d, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = (kernelSize, kernelSize)
        self.kernelSizeNum = kernelSize * kernelSize
        self.dilation = (dilation, dilation)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.weights = nn.Parameter(torch.Tensor(self.outChannels, self.inChannels, self.kernelSizeNum))
        
    def forwardProp(self, x):
        width = self.calculateWidth(x)
        height = self.calculateHeight(x)
        windows = self.calculateWindows(x)
        
        result = torch.zeros([x.shape[0] * self.outChannels, width, height], dtype=torch.float32, device=device)
        
        for c in range(x.shape[1]):
            for i in range(self.out_channels):
                x = torch.matmul(windows[c], self.weights[i][c]) 
                x = x.view(-1, width, height)
                result[i * x.shape[0]:(i + 1) * x.shape[0]] += x
        
        return result

    def calculateWindows(self, x):
        windows = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride)
        windows = windows.transpose(1, 2).contiguous().view(-1, x.shape[1], self.kernal_size_number)
        windows = windows.transpose(0, 1)

        return windows
    
    def calculateWidth(self, x):
        return ( (x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1

    def calculateHeight(self, x):
        return ( (x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)// self.stride[1]) + 1

    def getWeights(self):
        kernal_size = int(math.sqrt(self.kernal_size_number))
        return nn.Parameter(self.weights.view(self.out_channels, self.n_channels, kernal_size, kernal_size))

#---- functions ----
def cv2_to_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image)
    image = image.float() / torch.max(image)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    return image

def process_predictions(output, threshold=0.70):
    boxes = output[0]['boxes']
    labels = output[0]['labels']
    scores = output[0]['scores']
    predictions = []
    for idx in range(len(scores)):
        if scores[idx] >= threshold:
            predictions.append({
                'box': boxes[idx],
                'label': COCO_INSTANCE_CATEGORY_NAMES[labels[idx]],
                'score': scores[idx]
            })
    if len(predictions) == 0:
        idx = scores.index(max(scores))
        predictions.append({
            'box': boxes[idx],
            'label': COCO_INSTANCE_CATEGORY_NAMES[labels[idx]],
            'score': scores[idx]
        })
    return predictions

def draw_boxes(image, predictions):
    size_factor = min(image.shape[:-1]) // 500 + 1
    for obj in predictions:        
        image = cv2.rectangle(
            img = image,
            pt1 = (int(obj['box'][0].round()), int(obj['box'][1].round())),
            pt2 = (int(obj['box'][2].round()), int(obj['box'][3].round())),
            color = (0, 255, 0),
            thickness = size_factor
        )

        image = cv2.putText(
            img = image,
            text = obj['label'],
            org = (int(obj['box'][0]), int(obj['box'][1] - 5)),
            fontFace = cv2.FONT_HERSHEY_PLAIN,
            fontScale = size_factor,
            color = (0, 255, 0),
            thickness = size_factor
        )
    return image


#model = fasterrcnn_resnet50_fpn(pretrained=True)
model = maskrcnn_resnet50_fpn(pretrained=True)

# NEW CODE
model.backbone.fpn.add_module("4", CustomConv2d(20, 20, 5))

model.eval()
model.to(device)
img = cv2.imread("street_img.png", cv2.IMREAD_ANYCOLOR)

tensor = cv2_to_tensor(img)
tensor = tensor.to(device)              # placing the tensor on the gpu (to run it on the gpu)
pred = model(tensor)                    # detects the object
pred = process_predictions(pred)        # filters the predictions by confidence scores
img = draw_boxes(img, pred)             # draws the boxes around objects on the original image

img = cv2.resize(img, (1200, 800))
cv2.imshow("obj_detect", img)
cv2.waitKey(0)
