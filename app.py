from flask import Flask, request, jsonify
import torch
import model.model as module_arch  # Adjust import
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# Load your model (adjust as needed)
try:
    model = module_arch.DenseNet(num_classes=2)
    checkpoint = torch.load('saved/models/BCDensenet/0224_034642/model_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    try:
        image = request.files['image'].read()
        image = Image.open(io.BytesIO(image)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error':f'prediction error: {e}'})

if __name__ == '__main__':
    app.run(debug=True)
