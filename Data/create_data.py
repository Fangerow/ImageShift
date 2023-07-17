import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

from config import cfg


def shift_image(image_path, interpolation=cv2.INTER_CUBIC):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Unable to open image at {image_path}")
        return None

    matrix = np.float32([[1, 0, cfg['shift_value']],
                         [0, 1, cfg['shift_value']]])
    shifted = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                             flags=interpolation, borderMode=cv2.BORDER_WRAP)

    return shifted


def process_images(src_dir, target_dir):
    if not os.path.isdir(src_dir):
        print(f"{src_dir} is not a directory")
        return

    os.makedirs(target_dir, exist_ok=True)

    image_paths = [os.path.join(src_dir, img) for img in os.listdir(src_dir)
                   if os.path.isfile(os.path.join(src_dir, img))]

    for img_path in image_paths:
        shifted = shift_image(img_path)
        if shifted is not None:
            cv2.imwrite(os.path.join(target_dir, os.path.basename(img_path)), shifted)

    print(f"All images have been successfully shifted and saved in {target_dir}")


def create_dataset(src_dir, dst_dir, test_size=0.2):
    src_image_paths = [os.path.join(src_dir, img) for img in os.listdir(src_dir)
                       if os.path.isfile(os.path.join(src_dir, img))]
    dst_image_paths = [os.path.join(dst_dir, img) for img in os.listdir(dst_dir)
                       if os.path.isfile(os.path.join(dst_dir, img))]

    if len(src_image_paths) != len(dst_image_paths):
        print("The number of images in src_dir does not match the number of images in dst_dir")
        return

    image_pairs = list(zip(src_image_paths, dst_image_paths))
    train_pairs, test_pairs = train_test_split(image_pairs, test_size=test_size)

    return train_pairs, test_pairs

