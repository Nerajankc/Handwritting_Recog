import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from PIL import Image as im
import numpy as np
from segmentor import detect, prepare_img, sort_multiline
from path import Path
import cv2
from typing import List
import argparse


list_img_names_serial = []


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


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=Path('../data/page'))
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--sigma', type=float, default=11)
parser.add_argument('--theta', type=float, default=7)
parser.add_argument('--min_area', type=int, default=100)
parser.add_argument('--img_height', type=int, default=1000)
parsed = parser.parse_args()

print("File path: ", parsed.data)
def save_image_names_to_text_files(input_image, output_dir=None):
    """
    Process an input image, segment it into words, and save crops to the specified directory.
    
    Args:
        input_image: Path to the input image file
        output_dir: Directory to save cropped images (default: "../test_images")
        
    Returns:
        tuple: (output_directory, list of saved image paths)
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = "../test_images"
    
    # Clear the list if we're starting fresh
    list_img_names_serial.clear()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory Created: {output_dir}")

    img = prepare_img(cv2.imread(input_image), parsed.img_height)
    detections = detect(img,
                        kernel_size=parsed.kernel_size,
                        sigma=parsed.sigma,
                        theta=parsed.theta,
                        min_area=parsed.min_area)

    lines = sort_multiline(detections)
    saved_image_paths = []
    
    for line_idx, line in enumerate(lines):
        for word_idx, det in enumerate(line):
            xs = [det.bbox.x, det.bbox.x, det.bbox.x +
                  det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
            ys = [det.bbox.y, det.bbox.y + det.bbox.h,
                  det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
            print(det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h)
            crop_img = img[det.bbox.y:det.bbox.y +
                          det.bbox.h, det.bbox.x:det.bbox.x+det.bbox.w]

            filename = f"line{line_idx}word{word_idx}.jpg"
            file_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(file_path, crop_img)
            saved_image_paths.append(file_path)
            
            list_img_names_serial.append(filename)
            print(list_img_names_serial)

    # Save the list of filenames to a text file in the output directory
    textfile_path = os.path.join(output_dir, "img_names_sequence.txt")
    with open(textfile_path, "w") as textfile:
        for element in list_img_names_serial:
            textfile.write(element + "\n")
    
    return output_dir, saved_image_paths

if __name__ == "__main__":
    test_image = "../data/page/input.png"
    output_dir, image_paths = save_image_names_to_text_files(test_image)
    print(f"Images saved to: {output_dir}")
    print(f"Generated {len(image_paths)} images")
