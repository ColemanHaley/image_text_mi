import os
import cv2
import numpy as np
import argparse

from tqdm import tqdm

def load_images_from_folder(folder, size):
    images = []
    for filename in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Resize image to the specified size
            img = cv2.resize(img, (size, size))
            images.append(img)
    return images

def compute_average_image(images):
    # Check if there are any images
    if len(images) == 0:
        raise ValueError("No images to process")

    # Initialize an array with zeros
    avg_image = np.zeros_like(images[0], dtype=np.float64)

    # Sum all images
    for img in tqdm(images):
        avg_image += img

    # Compute the average
    avg_image /= len(images)

    # Convert to uint8 (standard image format)
    avg_image = np.uint8(avg_image)

    return avg_image

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compute the average image from a directory of images.')
    parser.add_argument('folder', type=str, help='Path to the directory containing images')
    parser.add_argument('output', type=str, help='Path to the output file')
    parser.add_argument('--size', type=int, default=224, help='Size to which images will be resized (default: 224)')
    
    args = parser.parse_args()
    
    # Load images from the folder
    images = load_images_from_folder(args.folder, args.size)

    # Compute the average image
    avg_image = compute_average_image(images)

    # Save the resulting average image
    cv2.imwrite(args.output, avg_image)


if __name__ == "__main__":
    main()
