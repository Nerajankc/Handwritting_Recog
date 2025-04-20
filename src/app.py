import streamlit as st
import os
import sys
import uuid
from PIL import Image
import numpy as np
import cv2

# Add src directory to path
sys.path.append('src')
import inference

# Set page configuration
st.set_page_config(
    page_title="Word Recognition OCR",
    page_icon="📝",
    layout="wide",
)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def main():
    st.title("Word Recognition OCR")
    st.write("Upload an image containing text to extract the content")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process button
        if st.button("Extract Text"):
            with st.spinner("Processing image..."):
                # Process the image using our inference module
                try:
                    result = inference.run_inference_on_image(file_path)
                    
                    with col2:
                        st.subheader("Recognized Text")
                        st.write(result["formatted_text"])
                    
                    # Display segmented words
                    st.subheader("Segmented Words")
                    output_dir = result["output_directory"]
                    
                    # Get all image files from the output directory
                    image_files = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    image_files = sorted(image_files, key=lambda x: x) # Sort by filename
                    
                    # Display images in a grid (up to 5 per row)
                    if image_files:
                        num_cols = min(5, len(image_files))
                        cols = st.columns(num_cols)
                        
                        for i, img_file in enumerate(image_files):
                            img_path = os.path.join(output_dir, img_file)
                            if os.path.exists(img_path) and img_file != "img_names_sequence.txt":
                                col_idx = i % num_cols
                                with cols[col_idx]:
                                    img = Image.open(img_path)
                                    st.image(img, caption=f"Word {i+1}")
                    
                    # Show individual word predictions
                    st.subheader("Individual Word Predictions")
                    for i, word in enumerate(result["prediction"]):
                        st.text(f"Word {i+1}: {word}")
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.exception(e)
            
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)

    # Add information about the app
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses OCR (Optical Character Recognition) to extract "
        "text from images. Upload an image containing text, and the app will "
        "process it and display the recognized text."
    )
    
    st.sidebar.title("Instructions")
    st.sidebar.markdown(
        """
        1. Upload an image using the file uploader
        2. Click 'Extract Text' to process the image
        3. View the recognized text and segmented words
        """
    )

if __name__ == "__main__":
    main() 