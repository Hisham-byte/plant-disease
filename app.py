import os
import re
import json
import torch
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, request, jsonify
from groq import Groq
import torch.nn as nn

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

groq_api_key = "_____"
client = Groq(api_key=groq_api_key)

def normalize_for_search(name):
    parts = name.split('___')
    disease_part = parts[-1]
    disease_part_clean = disease_part.replace('_', ' ')
    disease_part_clean = re.sub(r'\s+', ' ', disease_part_clean)
    return disease_part_clean.lower().strip()

class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64*28*28,512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512,K)
        )
    def forward(self,X):
        out = self.conv_layers(X)
        out = out.view(out.size(0),-1)
        out = self.dense_layers(out)
        return out

model = CNN(K=39)
model.load_state_dict(torch.load('/Users/sarthaklade/Downloads/plant_disease_cnn_model.pt', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  
])
            
data_dir = "/Users/sarthaklade/Downloads/Projects/DL_Project/Plant_leave_diseases_dataset_without_augmentation"
class_names = sorted(os.listdir(data_dir))
idx_to_class = {i: name for i, name in enumerate(class_names)}
            
@app.route('/')
def home():
    return render_template('index.html')
            
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image'] 
        image = Image.open(file.stream).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            _, pred = torch.max(output, 1)
            predicted_class = idx_to_class[pred.item()]

        normalized_pred = normalize_for_search(predicted_class)
    
        prompt = (
            f"Provide detailed information about {normalized_pred} plant disease in JSON format. "
            "The JSON must include:\n"
            "{\n"
            '  "description": {\n'
            '    "name": disease name,\n'
            '    "causes": cause of the disease,\n'
            '    "symptoms": detailed symptoms,\n'
            '    "affected_crops": affected plants,\n'
            '    "treatment": treatment methods\n'
            '  },\n'
            '  "possible_steps": [\n'   
            '    {"name": step name, "description": step description}\n'
            '  ],\n'
            '  "supplements": [\n'
            '    {"name": supplement name, "link": URL to product}\n'
            '  ]\n'
            "}\n"
            "Make sure to fill all fields with actual data, not empty strings, and provide supplements if available.\n"
            "For the supplements section, include relevant products that can help treat or prevent this disease.\n"
            "Format the output strictly as valid JSON without any markdown formatting."
        )
            
        chat_completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert plant pathologist. Provide accurate, detailed and practical information about plant diseases. Always provide complete information for every requested field."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=1500,
            temperature=0.2
        )

        response_text = chat_completion.choices[0].message.content
            
        response_text = re.sub(r"^```json\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)
        response_text = response_text.strip()
            
        try:
            disease_info = json.loads(response_text)
        except json.JSONDecodeError:
            disease_info = {
                "description": {
                    "name": normalized_pred,
                    "causes": "",
                    "symptoms": "",
                    "affected_crops": "",
                    "treatment": ""
                },
                "possible_steps": [],
                "supplements": []
            }
        desc = disease_info.get("description", {})
        description_text = (
            f"Name: {desc.get('name', '')}\n"
            f"Causes: {desc.get('causes', '')}\n"
            f"Symptoms: {desc.get('symptoms', '')}\n"
            f"Affected Crops: {desc.get('affected_crops', '')}\n"
            f"Treatment: {desc.get('treatment', '')}"
        )
        
        return jsonify({
            "disease_name": predicted_class,
            "description": description_text,
            "possible_steps": disease_info.get("possible_steps", []),
            "supplements": disease_info.get("supplements", [])
        })
            
    except Exception as e:  
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
