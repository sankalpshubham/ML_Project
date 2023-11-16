# from torchvision.io.image import read_image
# from torchvision.models.resnet import resnet18

# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained on COCO
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torch, cv2, base64
import numpy as np

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
#--------

#model = fasterrcnn_resnet50_fpn(pretrained=True)
model = maskrcnn_resnet50_fpn(pretrained=True)
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
