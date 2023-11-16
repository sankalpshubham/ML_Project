# Step 1: Import necessary libraries
import torch, cv2
import torchvision.transforms as transforms
from torchvision.models import detection
from PIL import Image, ImageDraw
import requests

# Step 2: Load the pre-trained YOLOv3 model
model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Step 3: Define the transformation for input images
transform = transforms.Compose([transforms.ToTensor()])

# Step 4: Load an image for testing
image_url = "https://www.adorama.com/alc/wp-content/uploads/2017/10/nyc-street-view-google-trusted-photographer-feature-1280x720.jpg"  # Replace with the URL or local path of your image
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# image = cv2.imread("street_img.png", cv2.IMREAD_ANYCOLOR)

# Step 5: Preprocess the image
input_image = transform(image)
input_batch = input_image.unsqueeze(0)  # Add a batch dimension

# Step 6: Run the image through the model for inference
with torch.no_grad():
    prediction = model(input_batch)

# Step 7: Get bounding box coordinates and labels
boxes = prediction[0]['boxes'].tolist()
labels = prediction[0]['labels'].tolist()

# Step 8: Draw bounding boxes on the image
draw = ImageDraw.Draw(image)
for box, label in zip(boxes, labels):
    draw.rectangle(box, outline="red", width=2)
    draw.text((box[0], box[1]), f"Label: {label}", fill="red")

# Step 9: Display the result
image.show()
