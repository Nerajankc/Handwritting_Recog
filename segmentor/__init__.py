from collections import defaultdict
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


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
