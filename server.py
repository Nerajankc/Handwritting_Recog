from flask import Flask, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
import sys

# Use absolute path for src directory
sys.path.append(os.path.abspath('src'))  # Add src directory to path
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
        result["uploaded_image"] = file_path
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({
            "error": str(e), 
            "details": error_details
        }), 500

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <head>
            <title>OCR Text Recognition</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                form { margin: 20px 0; }
                .instructions { background: #f8f8f8; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>OCR Text Recognition Service</h1>
            <div class="instructions">
                <p>Upload an image containing text to extract the content:</p>
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="image" accept="image/*">
                    <input type="submit" value="Extract Text">
                </form>
            </div>
            <p>API Usage with curl:</p>
            <pre>curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: multipart/form-data" \\
  -F "image=@/path/to/your/image.png"</pre>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 