"""
CS 4391 Homework 5 Programming
Implement the __getitem__() function in this python script
"""
import torch
import torch.utils.data as data
import csv
import os, math
import sys
import time
import random
import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# The dataset class
class CrackerBox(data.Dataset):
    def __init__(self, image_set = 'train', data_path = 'data2'):

        self.name = 'cracker_box_' + image_set
        self.image_set = image_set
        self.data_path = data_path
        self.classes = ('__background__', 'cracker_box')
        self.width = 640
        self.height = 480
        self.yolo_image_size = 448
        self.scale_width = self.yolo_image_size / self.width
        self.scale_height = self.yolo_image_size / self.height
        self.yolo_grid_num = 7
        self.yolo_grid_size = self.yolo_image_size / self.yolo_grid_num
        # split images into training set and validation set
        self.gt_files_train, self.gt_files_val = self.list_dataset()
        # the pixel mean for normalization
        self.pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)

        # training set
        if image_set == 'train':
            self.size = len(self.gt_files_train)
            self.gt_paths = self.gt_files_train
            print('%d images for training' % self.size)
        else:
            # validation set
            self.size = len(self.gt_files_val)
            self.gt_paths = self.gt_files_val
            print('%d images for validation' % self.size)


    # list the ground truth annotation files
    # use the first 100 images for training
    def list_dataset(self):
    
        filename = os.path.join(self.data_path, '*.txt')
        gt_files = sorted(glob.glob(filename))
        
        gt_files_train = gt_files[:100]
        gt_files_val = gt_files[100:110]
        
        return gt_files_train, gt_files_val


    # TODO: implement this function
    def __getitem__(self, idx):
    
        # gt file
        filename_gt = self.gt_paths[idx]
        
        ### ADD YOUR CODE HERE ###
        img_idx = os.path.splitext(os.path.basename(filename_gt))[0]
        img_file = os.path.join(self.data_path, f'{img_idx}.jpg')
        #print(img_idx, img_file)
        image = cv2.imread(img_file)
        image = cv2.resize(image, (self.yolo_image_size, self.yolo_image_size)).astype(np.float32)
        image = (image - self.pixel_mean) / 255  # subtract pixel mean
        image = image.transpose((2, 0, 1))       # HWC to CHW
        image_blob = torch.FloatTensor(image)
        
        #__________________________________________________________________
        # load ground truth bounding boxes
        with open(filename_gt, 'r') as file:
            lines = file.readlines()
        
        gt_box_blob = torch.zeros((5, self.yolo_grid_num, self.yolo_grid_num))
        gt_mask_blob = torch.zeros((self.yolo_grid_num, self.yolo_grid_num))
        class_label_blob = torch.zeros((1, self.yolo_grid_num, self.yolo_grid_num))


        for line in lines:
            bbox = list(map(float, line.split()))
        
        # calculating yolo format bounding boxes
            class_label = bbox[0]
            cx = bbox[1]
            cy = bbox[2]
            w = bbox[3]
            h = bbox[4]
        
        
            grid_x, grid_y = int(cx * self.yolo_grid_num), int(cy * self.yolo_grid_num)
            gt_box_blob[0, grid_y, grid_x] = cx * self.yolo_grid_num - grid_x
            gt_box_blob[1, grid_y, grid_x] = cy * self.yolo_grid_num - grid_y
            gt_box_blob[2, grid_y, grid_x] = w
            gt_box_blob[3, grid_y, grid_x] = h
            gt_box_blob[4, grid_y, grid_x] = 1
            class_label_blob[0, grid_y, grid_x] = class_label

            # creating ground truth mask
            gt_mask_blob[grid_y, grid_x] = 1

        gt_box_blob = torch.FloatTensor(gt_box_blob)
        gt_mask_blob = torch.FloatTensor(gt_mask_blob)
        class_label_blob = torch.FloatTensor(class_label_blob)
        
        # this is the sample dictionary to be returned from this function
        sample = {'image': image_blob,
                  'gt_box': gt_box_blob,
                  'gt_mask': gt_mask_blob,
                  'class_labels': class_label_blob}
        

        return sample

    # len of the dataset
    def __len__(self):
        return self.size
        

# draw grid on images for visualization
def draw_grid(image, line_space=64):
    H, W = image.shape[:2]
    image[0:H:line_space] = [255, 255, 0]
    image[:, 0:W:line_space] = [255, 255, 0]


# the main function for testing
if __name__ == '__main__':
    dataset_train = CrackerBox('train')
    dataset_val = CrackerBox('val')
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)
    
    # visualize the training data
    for i, sample in enumerate(train_loader):
        
        image = sample['image'][0].numpy().transpose((1, 2, 0))
        gt_box = sample['gt_box'][0].numpy()
        gt_mask = sample['gt_mask'][0].numpy()
        class_label = sample['class_labels'][0].numpy()

        #print(sample['class_labels'])

        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        im = image * 255.0 + dataset_train.pixel_mean
        im = im.astype(np.uint8)
        plt.imshow(im[:, :, (2, 1, 0)])
        plt.title('input image (448x448)', fontsize = 16)

        ax = fig.add_subplot(1, 3, 2)
        draw_grid(im)
        plt.imshow(im[:, :, (2, 1, 0)])

        indices = zip(*np.where(gt_mask == 1))
        for y, x in indices:
            cx = gt_box[0, y, x] * dataset_train.yolo_grid_size + x * dataset_train.yolo_grid_size
            cy = gt_box[1, y, x] * dataset_train.yolo_grid_size + y * dataset_train.yolo_grid_size
            w = gt_box[2, y, x] * dataset_train.yolo_image_size
            h = gt_box[3, y, x] * dataset_train.yolo_image_size

            x1 = cx - w * 0.5
            x2 = cx + w * 0.5
            y1 = cy - h * 0.5
            y2 = cy + h * 0.5

            #print("_")
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor="none")
            ax.add_patch(rect)
            plt.text((cx - w/2), (cy - h/2), str(class_label[0, y, x]), backgroundcolor='g')
            plt.plot(cx, cy, 'ro', markersize=12)

        #print(image.shape, gt_box.shape)
        plt.title('Ground truth bounding box in YOLO format', fontsize=16)

        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(gt_mask)
        plt.title('Ground truth mask in YOLO format (7x7)', fontsize=16)
        plt.show()