import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def train_maskrcnn():
    # Define your data transforms (you might need to adjust these)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # Load the COCO dataset
    train_dataset = CocoDetection(root='./coco_minitrain_25k', annFile='./coco_minitrain_25k/annotations/instances_train2017.json', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

    # Create the Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    num_classes = len(train_dataset.coco.getCatIds())  # Get the number of classes in the COCO dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, num_classes)

    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set the model to training mode
    model.train()

    # Define the optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

    # Save the trained model
    torch.save(model.state_dict(), 'maskrcnn_model.pth')

if __name__ == "__main__":
    train_maskrcnn()