# Handwriting Recognition Project

This project implements a handwriting recognition system using deep learning. It also includes a Flask-based web service for easy text extraction from images. The system is containerized using Docker for easy deployment and scalability.

## 🌟 Features

- Deep learning-based handwriting recognition
- RESTful API for text extraction from images
- Web interface for easy file uploads 
- Docker support for containerized deployment
- Multiple trained models available
- Support for image segmentation and processing

## 🛠️ Technology Stack

- Python 3.x
- TensorFlow/Keras for deep learning
- Flask for web service
- OpenCV for image processing
- Docker for containerization
- Streamlit for visualization (optional)

## 📋 Prerequisites

- Python 3.x
- Docker (optional)
- GPU support (recommended for training)

## 🚀 Getting Started

### Local Setup

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd handwriting-project
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the server:
   ```bash
   python server.py
   ```

### Docker Setup

1. Build the Docker image:
   ```bash
   docker-compose build
   ```

2. Run the container:
   ```bash
   docker-compose up
   ```

## 🔧 API Usage

### Endpoint: `/predict`

**Method:** POST

**Request:**
- Content-Type: multipart/form-data
- Body: image file

**Example using curl:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/your/image.png"
```

## 📁 Project Structure

```
.
├── src/                    # Source code directory
├── segmentor/              # Image segmentation modules
├── data/                   # Training data
├── test_images/           # Test images
├── uploads/               # Uploaded images storage
├── outputs/               # Model outputs
├── server.py              # Flask server
├── train_and_inference.ipynb  # Training notebook
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker compose configuration
└── requirements.txt       # Python dependencies
```

## 🏃‍♂️ Training

The model training process is documented in `train_and_inference.ipynb`. Multiple trained models are available:
- ocr_model_50_epoch.weights.h5
- ocr_model_v4.weights.h5
- ocr_model_v5.weights.h5
- ocr_model_v8.weights.h5

## 📝 Model Performance

Training results and model performance metrics can be found in `epochs_result.txt`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

Private 

## 👥 Authors

Nirajan KC, Ranjan Thakur

## 🙏 Acknowledgments

- [Add acknowledgments here]
