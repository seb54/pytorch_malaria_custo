import os
from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model_utils import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Charger le modèle
model = load_model('models/best_model_cnn.pth')
model.eval()  # Mettre explicitement en mode évaluation

# Définir la transformation pour les images
# Utilisation des mêmes paramètres que pendant l'entraînement
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # Utilisation des valeurs standard de normalisation ImageNet
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='Aucun fichier sélectionné')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='Aucun fichier sélectionné')

    try:
        # Charger et transformer l'image
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        # Prédiction avec torch.no_grad()
        with torch.no_grad():
            outputs = model(image_tensor)
            # Utiliser softmax pour obtenir des probabilités
            probabilities = F.softmax(outputs, dim=1)
            predicted_proba, predicted = torch.max(probabilities, 1)
            
        # Convertir la prédiction en étiquette
        classes = ['Parasité', 'Non parasité']
        result = classes[predicted.item()]
        confidence = predicted_proba.item() * 100  # Convertir en pourcentage
        
        return render_template(
            'result.html', 
            result=result, 
            confidence=f"{confidence:.2f}%",
            probabilities={
                'Parasité': f"{probabilities[0][0].item()*100:.2f}%",
                'Non parasité': f"{probabilities[0][1].item()*100:.2f}%"
            }
        )
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True) 