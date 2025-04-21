#!/usr/bin/env python3
"""
Handwriting Segmentation Tool

This script processes handwritten text images and segments them into individual words.
It saves the segmented words as separate image files and creates a sequence file
containing the order of the words.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2

# Add parent directory to Python path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from segmentor import HandwritingSegmentor

def retrieve_image_files(data_dir: Path) -> List[Path]:
    """Retrieve all image files from a specified directory.

    Args:
        data_dir: Path to the directory containing image files.

    Returns:
        A list of Paths to the image files found in the directory.
    """
    res = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res


class HandwritingProcessor:
    """Handles the processing of handwritten text images and saving of results."""
    
    def __init__(self, output_dir: str = "../output"):
        """
        Initialize the processor with output directory.
        
        Args:
            output_dir: Directory where segmented words will be saved
        """
        self.output_dir = Path(output_dir)
        self.segmentor = HandwritingSegmentor()
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the processor."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _ensure_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            self.logger.info(f"Created output directory: {self.output_dir}")
            
    def process_image(self, 
                     image_path: str,
                     target_height: int = 1000,
                     kernel_size: int = 25,
                     sigma: float = 11.0,
                     aspect_ratio: float = 7.0,
                     min_area: int = 100) -> Tuple[List[str], int]:
        """
        Process a single handwritten text image.
        
        Args:
            image_path: Path to the input image
            target_height: Height to resize the image to
            kernel_size: Size of the kernel for word detection
            sigma: Sigma value for Gaussian filter
            aspect_ratio: Expected width/height ratio of words
            min_area: Minimum area for word candidates
            
        Returns:
            Tuple containing:
                - List of paths to saved word images
                - Number of words detected
        """
        self._ensure_output_directory()
        
        # Read and preprocess image
        self.logger.info(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
            
        processed_image = self.segmentor.preprocess_image(image, target_height)
        
        # Detect and arrange words
        words = self.segmentor.detect_words(
            processed_image,
            kernel_size=kernel_size,
            sigma=sigma,
            aspect_ratio=aspect_ratio,
            min_area=min_area
        )
        
        lines = self.segmentor.arrange_words(words)
        
        # Save individual word images
        saved_paths = []
        sequence = []
        
        for line_idx, line in enumerate(lines):
            for word_idx, word in enumerate(line):
                filename = f"line{line_idx:03d}_word{word_idx:03d}.jpg"
                filepath = self.output_dir / filename
                
                cv2.imwrite(str(filepath), word.image)
                saved_paths.append(str(filepath))
                sequence.append(filename)
                
        # Save sequence file
        sequence_file = self.output_dir / "word_sequence.txt"
        with open(sequence_file, "w") as f:
            f.write("\n".join(sequence))
            
        self.logger.info(f"Processed {len(saved_paths)} words")
        return saved_paths, len(saved_paths)


def get_image_files(directory: str) -> List[Path]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Directory to search for images
        
    Returns:
        List of paths to image files
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    directory_path = Path(directory)
    
    return [
        f for f in directory_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Segment handwritten text into individual words"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='../data/page',
        help='Input directory containing images or path to single image'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../output',
        help='Output directory for segmented words'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=1000,
        help='Target height for image preprocessing'
    )
    parser.add_argument(
        '--kernel-size',
        type=int,
        default=25,
        help='Size of the filter kernel (odd integer)'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=11.0,
        help='Standard deviation for Gaussian filter'
    )
    parser.add_argument(
        '--aspect-ratio',
        type=float,
        default=7.0,
        help='Expected width/height ratio of words'
    )
    parser.add_argument(
        '--min-area',
        type=int,
        default=100,
        help='Minimum area for word candidates'
    )
    
    args = parser.parse_args()
    
    try:
        processor = HandwritingProcessor(args.output)
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Process single image
            saved_paths, word_count = processor.process_image(
                str(input_path),
                args.height,
                args.kernel_size,
                args.sigma,
                args.aspect_ratio,
                args.min_area
            )
            print(f"Processed 1 image, found {word_count} words")
            
        elif input_path.is_dir():
            # Process all images in directory
            image_files = get_image_files(input_path)
            total_words = 0
            
            for image_file in image_files:
                _, word_count = processor.process_image(
                    str(image_file),
                    args.height,
                    args.kernel_size,
                    args.sigma,
                    args.aspect_ratio,
                    args.min_area
                )
                total_words += word_count
                
            print(f"Processed {len(image_files)} images, found {total_words} words")
            
        else:
            print(f"Error: Input path {args.input} does not exist")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
