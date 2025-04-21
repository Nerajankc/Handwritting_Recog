from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class BoundingBox:
    """Represents a rectangular bounding box with position and dimensions."""
    x: int  # x-coordinate of top-left corner
    y: int  # y-coordinate of top-left corner
    width: int  # width of the box
    height: int  # height of the box

    @property
    def center(self) -> Tuple[float, float]:
        """Returns the center point (x, y) of the bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class WordSegment:
    """Represents a detected word segment in an image."""
    image: np.ndarray  # The cropped image containing the word
    bbox: BoundingBox  # The bounding box coordinates of the word


class HandwritingSegmentor:
    """A class for segmenting handwritten text into individual words."""

    @staticmethod
    def preprocess_image(image: np.ndarray, target_height: int) -> np.ndarray:
        """
        Preprocesses an input image for word segmentation.
        
        Args:
            image: Input image as numpy array (RGB or grayscale)
            target_height: Desired height of the output image
            
        Returns:
            Preprocessed grayscale image with specified height
        
        Raises:
            ValueError: If image dimensions or type are invalid
        """
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("Input must be a uint8 numpy array")
        if target_height <= 0:
            raise ValueError("Target height must be positive")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
            
        # Resize while maintaining aspect ratio
        aspect_ratio = target_height / gray_image.shape[0]
        new_width = int(gray_image.shape[1] * aspect_ratio)
        return cv2.resize(gray_image, (new_width, target_height))

    @staticmethod
    def _create_anisotropic_kernel(
        kernel_size: int,
        sigma: float,
        aspect_ratio: float
    ) -> np.ndarray:
        """
        Creates an anisotropic Gaussian kernel for word detection.
        
        Args:
            kernel_size: Size of the kernel (must be odd)
            sigma: Base standard deviation for Gaussian
            aspect_ratio: Width to height ratio for words
            
        Returns:
            2D numpy array containing the kernel values
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
            
        half_size = kernel_size // 2
        x_coords, y_coords = np.meshgrid(
            np.linspace(-half_size, half_size, kernel_size),
            np.linspace(-half_size, half_size, kernel_size)
        )
        
        # Compute anisotropic sigmas
        sigma_y = sigma
        sigma_x = sigma * aspect_ratio
        
        # Calculate kernel components
        gaussian = np.exp(-x_coords**2 / (2 * sigma_x) - y_coords**2 / (2 * sigma_y))
        x_component = (x_coords**2 - sigma_x**2) / (2 * np.pi * sigma_x**5 * sigma_y)
        y_component = (y_coords**2 - sigma_y**2) / (2 * np.pi * sigma_y**5 * sigma_x)
        
        kernel = (x_component + y_component) * gaussian
        return kernel / np.sum(kernel)

    def detect_words(
        self,
        image: np.ndarray,
        kernel_size: int = 25,
        sigma: float = 11,
        aspect_ratio: float = 7,
        min_area: int = 100
    ) -> List[WordSegment]:
        """
        Detects and segments individual words in the image.
        
        Args:
            image: Grayscale input image
            kernel_size: Size of the filter kernel (odd integer)
            sigma: Standard deviation for Gaussian filter
            aspect_ratio: Expected width/height ratio of words
            min_area: Minimum area for word candidates
            
        Returns:
            List of WordSegment objects containing detected words
        """
        if len(image.shape) != 2 or image.dtype != np.uint8:
            raise ValueError("Input must be a grayscale uint8 image")

        # Apply anisotropic filter
        kernel = self._create_anisotropic_kernel(kernel_size, sigma, aspect_ratio)
        filtered = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        
        # Threshold the filtered image
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = 255 - binary  # Invert for contour detection
        
        # Find and process contours
        contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        word_segments = []
        
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            word_image = image[y:y+h, x:x+w]
            word_segments.append(WordSegment(
                image=word_image,
                bbox=BoundingBox(x, y, w, h)
            ))
            
        return word_segments

    def arrange_words(
        self,
        words: List[WordSegment],
        max_line_gap: float = 0.7,
        min_words_per_line: int = 2
    ) -> List[List[WordSegment]]:
        """
        Arranges detected words into lines and sorts them.
        
        Args:
            words: List of detected word segments
            max_line_gap: Maximum vertical gap between words in the same line
            min_words_per_line: Minimum number of words to form a line
            
        Returns:
            List of lines, where each line is a list of words sorted by x-coordinate
        """
        if not words:
            return []
            
        # Calculate pairwise vertical IoU distances
        num_words = len(words)
        distances = np.ones((num_words, num_words))
        
        for i in range(num_words):
            for j in range(i, num_words):
                box1, box2 = words[i].bbox, words[j].bbox
                
                # Skip if boxes don't overlap vertically
                if (box1.y > box2.y + box2.height or 
                    box2.y > box1.y + box1.height):
                    continue
                    
                # Calculate vertical IoU
                intersection = min(box1.y + box1.height, box2.y + box2.height) - \
                             max(box1.y, box2.y)
                union = box1.height + box2.height - intersection
                iou = intersection / union if union > 0 else 0
                
                # Store Jaccard distance (1 - IoU)
                distances[i, j] = distances[j, i] = 1 - np.clip(iou, 0, 1)
        
        # Cluster words into lines using DBSCAN
        clusters = DBSCAN(
            eps=max_line_gap,
            min_samples=min_words_per_line,
            metric='precomputed'
        ).fit(distances)
        
        # Group words by line
        lines = defaultdict(list)
        for word_idx, cluster_id in enumerate(clusters.labels_):
            if cluster_id != -1:  # Skip noise points
                lines[cluster_id].append(words[word_idx])
        
        # Sort lines by vertical position and words within lines by horizontal position
        sorted_lines = []
        for line in sorted(lines.values(), 
                         key=lambda x: sum(w.bbox.center[1] for w in x) / len(x)):
            sorted_words = sorted(line, key=lambda w: w.bbox.center[0])
            sorted_lines.append(sorted_words)
            
        return sorted_lines
