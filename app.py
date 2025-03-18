import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, render_template
from PIL import Image

# Define the Model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Adjust this based on image size
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)  # 4 Classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Model
model = CustomCNN()
model.load_state_dict(torch.load("models/cancer_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Class Labels
class_labels = ["Benign", "Malignant Early Pre-B", "Malignant Pre-B", "Malignant Pro-B"]

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        file = request.files.get("file")  # Get uploaded file
        if file:
            img = Image.open(file).convert("RGB")
            img = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                output = model(img)
                prediction = torch.argmax(output, 1).item()
            
            result = class_labels[prediction]
            return render_template("result.html", prediction=result)  # Redirect to result page

    return render_template("index.html")  # Show upload form for GET requests

@app.route('/result')
def result():
    prediction = request.args.get("prediction", "No result")
    return render_template("result.html", prediction=prediction)

@app.route('/precautions')
def precautions():
    return render_template("pre.html")  # Ensure pre.html exists

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)
