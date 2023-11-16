# import os
# import subprocess
# import wget

# # Create directories to store the dataset
# os.makedirs("data/coco/train", exist_ok=True)
# os.makedirs("data/coco/val", exist_ok=True)


# # Download train2017 images
# train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
# wget.download(train_images_url, "train2017.zip")

# # Download val2017 images
# val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
# wget.download(val_images_url, "val2017.zip")

# # Download annotations for train2017 and val2017
# annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# wget.download(annotations_url, "annotations_trainval2017.zip")

# # Extract the train2017 images
# import zipfile
# with zipfile.ZipFile("train2017.zip", "r") as zip_ref:
#     zip_ref.extractall("data/coco/train")

# # Extract the val2017 images
# with zipfile.ZipFile("val2017.zip", "r") as zip_ref:
#     zip_ref.extractall("data/coco/val")

# # Extract the annotations
# with zipfile.ZipFile("annotations_trainval2017.zip", "r") as zip_ref:
#     zip_ref.extractall("data/coco/annotations")

import os
import subprocess
import wget
import random
import zipfile

# Define the desired sample size for train and val sets
train_sample_size = 1000  # Adjust this value to control the number of train images
val_sample_size = 200  # Adjust this value to control the number of val images

# Create directories to store the dataset
os.makedirs("data/coco/train", exist_ok=True)
os.makedirs("data/coco/val", exist_ok=True)
os.makedirs("data/coco/annotations/train", exist_ok=True)
os.makedirs("data/coco/annotations/val", exist_ok=True)

# # Download train2017 images
# train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
# wget.download(train_images_url, "train2017.zip")

# # Download val2017 images
# val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
# wget.download(val_images_url, "val2017.zip")

# Download annotations for train2017 and val2017
annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
wget.download(annotations_url, "annotations_trainval2017.zip")

# # Extract train2017 images
# with zipfile.ZipFile("train2017.zip", "r") as zip_ref:
#     # Select a random sample of image files
#     train_image_files = [f for f in zip_ref.namelist() if f.endswith(".jpg")]
#     random.shuffle(train_image_files)
#     selected_train_images = train_image_files[:train_sample_size]

#     # Extract the selected train images
#     zip_ref.extractall("data/coco/train", selected_train_images)

# # Extract val2017 images
# with zipfile.ZipFile("val2017.zip", "r") as zip_ref:
#     # Select a random sample of image files
#     val_image_files = [f for f in zip_ref.namelist() if f.endswith(".jpg")]
#     random.shuffle(val_image_files)
#     selected_val_images = val_image_files[:val_sample_size]

#     # Extract the selected val images
#     zip_ref.extractall("data/coco/val", selected_val_images)

# Extract annotations
with zipfile.ZipFile("annotations_trainval2017.zip", "r") as zip_ref:
    # Extract all annotations related to the selected train images
    train_annotations = [f for f in zip_ref.namelist() if f.startswith("instances_train2017")]
    zip_ref.extractall("data/coco/annotations/train", train_annotations)

    # Extract all annotations related to the selected val images
    val_annotations = [f for f in zip_ref.namelist() if f.startswith("instances_val2017")]
    zip_ref.extractall("data/coco/annotations/val", val_annotations)

