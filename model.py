"""
CS 4391 Homework 5 Programming
Implement the create_modules() function in this python script
"""
import os
import math
import torch
import torch.nn as nn
import numpy as np


# the YOLO network class
class YOLO(nn.Module):
    def __init__(self, num_boxes, num_classes):
        super(YOLO, self).__init__()
        # number of bounding boxes per cell (2 in our case)
        self.num_boxes = num_boxes
        # number of classes for detection (1 in our case: cracker box)
        self.num_classes = num_classes
        self.image_size = 448
        self.grid_size = 64
        # create the network
        self.network = self.create_modules()
        
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    #TODO: implement this function to build the network
    def create_modules(self):
        modules = nn.Sequential()

        ### ADD YOUR CODE HERE ###
        # hint: use the modules.add_module()
        modules.add_module("conv_1", torch.nn.Conv2d(3, 16, 3, 1, 1))
        modules.add_module("relu_1", torch.nn.ReLU())
        modules.add_module("maxpool_1", torch.nn.MaxPool2d(2, 2))

        modules.add_module("conv_2", torch.nn.Conv2d(16, 32, 3, 1, 1))
        modules.add_module("relu_2", torch.nn.ReLU())
        modules.add_module("maxpool_2", torch.nn.MaxPool2d(2, 2))

        modules.add_module("conv_3", torch.nn.Conv2d(32, 64, 3, 1, 1))
        modules.add_module("relu_3", torch.nn.ReLU())
        modules.add_module("maxpool_3", torch.nn.MaxPool2d(2, 2))

        modules.add_module("conv_4", torch.nn.Conv2d(64, 128, 3, 1, 1))
        modules.add_module("relu_4", torch.nn.ReLU())
        modules.add_module("maxpool_4", torch.nn.MaxPool2d(2, 2))

        modules.add_module("conv_5", torch.nn.Conv2d(128, 256, 3, 1, 1))
        modules.add_module("relu_5", torch.nn.ReLU())
        modules.add_module("maxpool_5", torch.nn.MaxPool2d(2, 2))

        modules.add_module("conv_6", torch.nn.Conv2d(256, 512, 3, 1, 1))
        modules.add_module("relu_6", torch.nn.ReLU())
        modules.add_module("maxpool_6", torch.nn.MaxPool2d(2, 2))

        modules.add_module("conv_7", torch.nn.Conv2d(512, 1024, 3, 1, 1))
        modules.add_module("relu_7", torch.nn.ReLU())
        
        modules.add_module("conv_8", torch.nn.Conv2d(1024, 1024, 3, 1, 1))
        modules.add_module("relu_8", torch.nn.ReLU())

        modules.add_module("conv_9", torch.nn.Conv2d(1024, 1024, 3, 1, 1))
        modules.add_module("relu_9", torch.nn.ReLU())

        modules.add_module("flatten", torch.nn.Flatten(1, -1))
        modules.add_module("fc1", torch.nn.Linear(50176, 256, True))
        modules.add_module("fc2", torch.nn.Linear(256, 256, True))
        modules.add_module("output", torch.nn.Linear(256, 539, True))
        modules.add_module("sigmoid", torch.nn.Sigmoid())

        return modules


    # output (batch_size, 5*B + C, 7, 7)
    # In the network output (cx, cy, w, h) are normalized to be [0, 1]
    # This function undo the noramlization to obtain the bounding boxes in the orignial image space
    def transform_predictions(self, output):
        batch_size = output.shape[0]
        x = torch.linspace(0, 384, steps=7)
        y = torch.linspace(0, 384, steps=7)
        corner_x, corner_y = torch.meshgrid(x, y, indexing='xy')
        corner_x = torch.unsqueeze(corner_x, dim=0)
        corner_y = torch.unsqueeze(corner_y, dim=0)
        corners = torch.cat((corner_x, corner_y), dim=0)
        # corners are top-left corners for each cell in the grid
        #___________________________________________________
        #corners = corners.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        corners = corners.to(output.device).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        #________________________________________________________
        pred_box = output.clone()

        # for each bounding box
        for i in range(self.num_boxes):
            # x and y
            pred_box[:, i*5, :, :] = corners[:, 0, :, :] + output[:, i*5, :, :] * self.grid_size
            pred_box[:, i*5+1, :, :] = corners[:, 1, :, :] + output[:, i*5+1, :, :] * self.grid_size
            # w and h
            pred_box[:, i*5+2, :, :] = output[:, i*5+2, :, :] * self.image_size
            pred_box[:, i*5+3, :, :] = output[:, i*5+3, :, :] * self.image_size

        return pred_box


    # forward pass of the YOLO network
    def forward(self, x):
        # raw output from the network
        output = self.network(x).reshape((-1, self.num_boxes * 5 + self.num_classes, 7, 7))
        # compute bounding boxes in the original image space
        pred_box = self.transform_predictions(output)
        return output, pred_box


# run this main function for testing
if __name__ == '__main__':
    network = YOLO(num_boxes=2, num_classes=1)
    print(network)

    image = np.random.uniform(-0.5, 0.5, size=(1, 3, 448, 448)).astype(np.float32)
    image_tensor = torch.from_numpy(image)
    print('input image:', image_tensor.shape)

    output, pred_box = network(image_tensor)
    print('network output:', output.shape, pred_box.shape)
    print(output)
