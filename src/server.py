from flask import Flask, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
import sys
sys.path.append('src')  # Add src directory to path if needed
import inference

app = Flask(__name__)

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    # If user does not select file, browser might
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Generate unique filename to avoid overwrites
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save the uploaded file
        file.save(file_path)
        
        # Use inference module to get text from image
        result = inference.run_inference_on_image(file_path)
        
        # Add saved path to result
        result["saved_path"] = file_path
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 