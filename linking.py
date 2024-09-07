from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import shutil
import subprocess
import json
from collections import Counter
import itertools

########################################
import torch
import re
import itertools
import json
import os
from collections import Counter
from pprint import pprint
import yaml
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import time
import h5py
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
import runmodel

def store_image(image): ## changeddd
     output_path = './data/images/val'
     new_name = 'VizWiz_demo_00000000.jpg'
     # Determine the new file path
     new_file_path = os.path.join(output_path, new_name)
     # Copy and rename the file
     shutil.copy(image, new_file_path)  

def write_question(question):
    file_path = './data/annotations/demo.json'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Update the JSON content with the new question
    if isinstance(data, list) and len(data) > 0 and 'question' in data[0]:
        data[0]['question'] = question
    else:
        print("Invalid JSON structure")
        return
    
    # Write the updated JSON content back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Question '{question}' has been written to '{file_path}'.")




#####################3

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        store_image(input_path)
        #print(input_path)
        runmodel.optimize()
        
        demo_json_path = os.path.join('predictions', 'demo.json')
        
        if not os.path.exists(demo_json_path):
            return jsonify({"error": "Prediction result not found"}), 500
        
        with open(demo_json_path, 'r') as json_file:
            demo_data = json.load(json_file)
        
        if not demo_data:
            return jsonify({"error": "Prediction data is empty"}), 500
        
        answer = demo_data[0].get('answer', 'No answer found')
        
        return jsonify({"message": "Image uploaded successfully", "filename": filename, "answer": answer}), 201
@app.route("/upload_text", methods=["POST"])
def upload_text():
    global question
    if 'text' not in request.form:
        return jsonify({"error": "No text part in the request"}), 400

    text = request.form['text']
    write_question(text)
    return jsonify({"message": "Text uploaded successfully", "text": text}), 201

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)