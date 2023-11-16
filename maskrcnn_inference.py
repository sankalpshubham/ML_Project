import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T

def inference_maskrcnn(model_path="maskrcnn_model.pth"):
    # Load the Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=False)
    num_classes = 80  # Update with the correct number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the transformation to be applied to the camera input
    transform = T.Compose([T.ToTensor()])

    # Open a connection to the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    with torch.no_grad():
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()

            # Transform the frame
            input_tensor = transform(frame).unsqueeze(0).to(device)

            # Perform inference
            predictions = model(input_tensor)

            # Post-process the predictions (you may need to customize this based on your model output)
            # For example, drawing bounding boxes on the image
            # ...

            # Display the result
            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference_maskrcnn()
