import os
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import cv2

from skimage import filters, transform

def image_preprocess(img):
    img = remove_background(img=img)
    img = remove_border(img=img)
    return img

def augment_image(img, min_change_ratio=0, rotate_degree=0, shear_degree=0, morphological_type=None, morphological_prob=0.5):
    if min_change_ratio > 0 and np.random.rand() > 0.5:
        img = change_aspect_ratio_with_padding(img, min_change_ratio=min_change_ratio)
    if np.random.rand() < morphological_prob:
        img = morphological_transform(img, kernel_dim=np.random.randint(2,4), type=morphological_type)
    img = affine_transform(img, rotate_degree, shear_degree)
    return img

def affine_transform(img, rotate_degree, shear_degree):
    height, width = img.shape[:2]

    # SHEAR
    shear_factor1 = np.tan(np.radians(random.uniform(-shear_degree, shear_degree)))
    shear_factor2 = np.tan(np.radians(random.uniform(-shear_degree, shear_degree)))
    new_w = int(width + height * abs(shear_factor1))
    new_h = int(height + width * abs(shear_factor2))
    S = np.eye(3)
    S[0, 1] = shear_factor1
    S[1, 0] = shear_factor2
    if shear_factor1 < 0:
        S[0, 2] = height * abs(shear_factor1)
    if shear_factor2 < 0:
        S[1, 2] = width * abs(shear_factor2)
    img = cv2.warpAffine(img, S[:2], (new_w, new_h), borderValue=(0, 0, 0))

    # ROTATE
    height, width = img.shape[:2]
    center_x, center_y = width // 2, height // 2
    rotate_degree = random.uniform(-rotate_degree, rotate_degree)
    rad = np.radians(rotate_degree)
    sin_rad = abs(np.sin(rad))
    cos_rad = abs(np.cos(rad))
    new_w = int(height * sin_rad + width * cos_rad)
    new_h = int(height * cos_rad + width * sin_rad)
    R = cv2.getRotationMatrix2D((center_x, center_y), rotate_degree, 1)
    R[0, 2] += (new_w / 2) - center_x
    R[1, 2] += (new_h / 2) - center_y
    img = cv2.warpAffine(img, R, (new_w, new_h), borderValue=(0, 0, 0))

    return img

def morphological_transform(img, kernel_dim, type):
    kernel = np.ones((kernel_dim, kernel_dim),np.uint8)
    if type == "erode":
        img = cv2.erode(img, kernel, iterations=1)
    elif type == "dilate":
        img = cv2.dilate(img, kernel, iterations=1)
    return img

def remove_background(img):
    img = img.astype(np.uint8)
    threshold = filters.threshold_otsu(img)
    img[img > threshold] = 255

    return img

def remove_border(img):
    threshold = filters.threshold_otsu(img)
    binarized_image = img > threshold
    r, c = np.where(binarized_image == 0)
    # r_center = int(r.mean() - r.min())
    # c_center = int(c.mean() - c.min())

    # Crop the img with a tight box
    img = img[r.min(): r.max(), c.min(): c.max()]
    return img

def resize_with_padding(img, img_size):
    if len(img.shape) == 3:
        H, W, _ = img.shape
    else:
        H, W = img.shape
    h, w = img_size

    rh, rw = h / H, w / W

    if rh > rw:
        new_h, new_w = int(H * rw) , w
    else:
        new_h, new_w = h, int(W* rh)
    
    img = cv2.resize(img, (new_w, new_h))
    new_img = np.ones((h, w), dtype=np.uint8) * 255
    
    x_start = int((h - new_h) / 2)
    y_start = int((w - new_w) / 2)
    new_img[x_start:(x_start + new_h), y_start:(y_start + new_w)] = img
    
    return new_img

def change_aspect_ratio_with_padding(img, min_change_ratio=0.05):
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    original_aspect_ratio = width / height
    if np.random.rand() > 0.5:
        new_aspect_ratio = original_aspect_ratio * np.random.uniform(0.8 - min_change_ratio, 1 - min_change_ratio)
    else:
        new_aspect_ratio = original_aspect_ratio * np.random.uniform(1 + min_change_ratio, 1.2 + min_change_ratio)

    if original_aspect_ratio > new_aspect_ratio:
        new_width = int(height * new_aspect_ratio)
        resized_image = cv2.resize(img, (new_width, height))
        padding1 = (width - new_width) // 2  # The amount of padding to add on each side
        padding2 = width - new_width - padding1
        padded_image = cv2.copyMakeBorder(resized_image, 0, 0, padding1, padding2, cv2.BORDER_CONSTANT, value=0)
    elif original_aspect_ratio < new_aspect_ratio:
        new_height = int(width / new_aspect_ratio)
        resized_image = cv2.resize(img, (width, new_height))
        padding1 = (height - new_height) // 2  # The amount of padding to add on each side
        padding2 = height - new_height - padding1
        padded_image = cv2.copyMakeBorder(resized_image, padding1, padding2, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        padded_image = img.copy()
    return padded_image

def add_noise(img, noise_ratio):
    height, width = img.shape

    num_noisy_pixels = int(noise_ratio * height * width)
    noise_index = np.random.choice(height * width, num_noisy_pixels, replace=False)
    noise_values = np.random.randint(0, 256, num_noisy_pixels)

    noisy_image = np.copy(img).flatten()
    noisy_image[noise_index] = noise_values
    noisy_image = noisy_image.reshape(height, width)

    return noisy_image

if __name__ == "__main__":
    img = cv2.imread("../../Dataset/VTCC_15_05/Scan_PNG_1im_filtered_fitted_15_05/id1/real/id1_real_0.png", 0)
    # new_img = change_aspect_ratio_with_padding(img, min_change_ratio=0.05)
    # new_img = rotate_image(img, np.random.randint(10, 20))
    img1 = 255 - image_preprocess(img)
    # img2 = augment_image(img1, rotate_degree=10, min_change_ratio=0, shear_degree=10, morphological_type="dilate")

    kernel = np.ones((2, 2),np.uint8)
    img2 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)

    # print(f"Original shape: {img.shape}")
    # print(f"New shape: {new_img.shape}") 

    cv2.imwrite("../0.png", 255-img)
    cv2.imwrite("../1.png", img1)
    cv2.imwrite("../2.png", img2)

# 30 epoch, lr = 1e-4, bs 64, cosine annual scheduler