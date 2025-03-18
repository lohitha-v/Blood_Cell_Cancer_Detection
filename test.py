import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
from app import CustomCNN, class_labels  # Import model & labels from app.py

# Load Model
model = CustomCNN()
model.load_state_dict(torch.load("models/cancer_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load and Predict
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(output, 1).item()
    
    return class_labels[prediction]

# Run Prediction from Command Line
if __name__ == "__main__":
    image_path = sys.argv[1]
    result = predict(image_path)
    print(f"Prediction: {result}")
