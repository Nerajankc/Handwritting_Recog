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
=======
@dataclass
class DetectorRes:
    img: np.ndarray
    bbox: BBox


def segment_words_in_image(img: np.ndarray,
                        kernel_size: int,
                        sigma: float,
                        theta: float,
                        min_area: int) -> List[DetectorRes]:
    """Segment words in a grayscale image using a scale-space technique.

    Args:
        img: A grayscale uint8 image.
        kernel_size: The size of the filter kernel, must be an odd integer.
        sigma: Standard deviation of Gaussian function used for filter kernel.
        theta: Approximated width/height ratio of words, filter function is distorted by this factor.
        min_area: Ignore word candidates smaller than specified area.

    Returns:
        List of DetectorRes instances, each containing the bounding box and the word image.
    """
    assert img.ndim == 2
    assert img.dtype == np.uint8

    # apply filter kernel
    kernel = compute_anisotropic_filter_kernel(kernel_size, sigma, theta)
    img_filtered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    img_thres = 255 - cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # append components to result
    res = []
    components = cv2.findContours(img_thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    print(components)
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < min_area:
            continue
        # append bounding box and image of word to result list
        x, y, w, h = cv2.boundingRect(c)  # bounding box as tuple (x, y, w, h)
        crop = img[y:y + h, x:x + w]
        res.append(DetectorRes(crop, BBox(x, y, w, h)))

    return res


def compute_anisotropic_filter_kernel(kernel_size: int,
                                       sigma: float,
                                       theta: float) -> np.ndarray:
    """Compute an anisotropic filter kernel for image processing.

    Args:
        kernel_size: Size of the kernel, must be odd.
        sigma: Standard deviation for the Gaussian function.
        theta: Ratio of width to height for the filter.

    Returns:
        Anisotropic filter kernel as a numpy array.
    """

    assert kernel_size % 2  # must be odd size

    # create coordinate grid
    half_size = kernel_size // 2
    xs = ys = np.linspace(-half_size, half_size, kernel_size)
    x, y = np.meshgrid(xs, ys)

    # compute sigma values in x and y direction, where theta is roughly the average x/y ratio of words
    sigma_y = sigma
    sigma_x = sigma_y * theta

    # compute terms and combine them
    exp_term = np.exp(-x ** 2 / (2 * sigma_x) - y ** 2 / (2 * sigma_y))
    x_term = (x ** 2 - sigma_x ** 2) / (2 * np.pi * sigma_x ** 5 * sigma_y)
    y_term = (y ** 2 - sigma_y ** 2) / (2 * np.pi * sigma_y ** 5 * sigma_x)
    kernel = (x_term + y_term) * exp_term

    # normalize and return kernel
    kernel = kernel / np.sum(kernel)
    return kernel


def convert_and_resize_image(img: np.ndarray, height: int) -> np.ndarray:
    """Convert image to grayscale and resize to the specified height.

    Args:
        img: Input image as a numpy array, can be grayscale or color.
        height: Desired height to resize the image to.

    Returns:
        Resized grayscale image as a numpy array.
    """
    assert img.ndim in (2, 3), "Image must be 2D (grayscale) or 3D (color)."
    assert height > 0, "Height must be a positive integer."
    assert img.dtype == np.uint8, "Image dtype must be uint8."
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if image is color
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def cluster_detections_into_lines(detections: List[DetectorRes],
                                   max_dist: float = 0.7,
                                   min_words_per_line: int = 2) -> List[List[DetectorRes]]:
    """Cluster word detections into lines based on Jaccard distance.

    Args:
        detections: List of word detections.
        max_dist: Maximum Jaccard distance for clustering.
        min_words_per_line: Minimum number of words per line to be considered valid.

    Returns:
        List of clustered lines, each containing word detections.
    """
    # compute matrix containing Jaccard distances (which is a proper metric)
    num_bboxes = len(detections)
    dist_mat = np.ones((num_bboxes, num_bboxes))
    for i in range(num_bboxes):
        for j in range(i, num_bboxes):
            a = detections[i].bbox
            b = detections[j].bbox
            if a.y > b.y + b.h or b.y > a.y + a.h:
                continue
            intersection = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
            union = a.h + b.h - intersection
            iou = np.clip(intersection / union if union > 0 else 0, 0, 1)
            dist_mat[i, j] = dist_mat[j, i] = 1 - iou  # Jaccard distance is defined as 1-iou

    dbscan = DBSCAN(eps=max_dist, min_samples=min_words_per_line, metric='precomputed').fit(dist_mat)

    clustered = defaultdict(list)
    for i, cluster_id in enumerate(dbscan.labels_):
        if cluster_id == -1:
            continue
        clustered[cluster_id].append(detections[i])

    res = sorted(clustered.values(), key=lambda line: [det.bbox.y + det.bbox.h / 2 for det in line])
    return res


def sort_detections_into_multiline(detections: List[DetectorRes],
                                    max_dist: float = 0.7,
                                    min_words_per_line: int = 2) -> List[List[DetectorRes]]:
    """Sort detections into multiple lines based on their x-coordinates.

    Args:
        detections: List of detections.
        max_dist: Maximum Jaccard distance for clustering.
        min_words_per_line: Minimum number of words per line to be considered valid.

    Returns:
        List of sorted lines, each line itself a list of detections.
    """
    lines = cluster_detections_into_lines(detections, max_dist, min_words_per_line)
    res = []
    for line in lines:
        res += sort_detections_by_x_coordinate(line)
    return res


def sort_detections_by_x_coordinate(detections: List[DetectorRes]) -> List[List[DetectorRes]]:
    """Sort detections by the x-coordinate of their bounding box centers.

    Args:
        detections: List of word detections.

    Returns:
        List of detections sorted by x-coordinate.
    """
    return [sorted(detections, key=lambda det: det.bbox.x + det.bbox.w / 2)]

