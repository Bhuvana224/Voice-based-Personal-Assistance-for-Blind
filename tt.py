from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables to store the latest text and image name
question = ""
imgname = ""

@app.route("/upload_image", methods=["POST"])
def upload_image():
    global imgname
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        imgname = filename
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
    question = text
    print(question)
    return jsonify({"message": "Text uploaded successfully", "text": text}), 201

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
    
