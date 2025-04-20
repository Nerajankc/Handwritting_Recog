FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for uploads and outputs
RUN mkdir -p uploads
RUN mkdir -p outputs

# Expose port for Flask and Streamlit
EXPOSE 5000 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application (can be overridden in docker-compose)
CMD ["python", "server.py"] 