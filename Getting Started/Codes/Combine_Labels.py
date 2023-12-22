# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 00:28:49 2023

@author: Novruz Mammadli
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm


def get_image_paths(input_folder):
    image_paths = []
    subfolder_names = []

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
                subfolder_names.append(os.path.basename(os.path.dirname(os.path.join(root, file))))

    return image_paths, subfolder_names


# Create the output folder if it doesn't exist
def creat_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def combine_labels(ship_folder, sea_folder, sky_folder, output_folder, label_colors, label_priorities, color_mask=False, save=True, SIZE_X=256, SIZE_Y=256):
    """
    *This function combines all separate masks together*
    Input:
        ship_folder: path of ship masks - text
        sea_folder: path of sea masks - text
        sky_folder: path of sky masks - text
        output_folder: output path wehere to save combined masks - text
        label_colors: if you want to save masks colorful you assign colors - dict
        label_priorities: binary values of combined masks - dict
        color_mask: if you want to save the masks colorful - Bool
        save:if you want to save the combined masks in output folder - Bool
        SIZE_X: width of images - int
        SIZE_Y: height of images - int
    Output:
        combined_images: is combined masks - numpy array
    """
    combined_images = []
    # Loop through the images in the ship folder
    for image_name in os.listdir(ship_folder):
        image_path = os.path.join(ship_folder, image_name)
        ship_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
        # Initialize the combined image with ship labels
        combined_image = np.zeros_like(ship_image, dtype=np.uint8)
    
        # Set ship labels to 2
        combined_image[ship_image > 120] = 2
    
        # Check if sea labels exist and overlay with priority
        sea_image_path = os.path.join(sea_folder, image_name)
        if os.path.exists(sea_image_path):
            sea_image = cv2.imread(sea_image_path, cv2.IMREAD_GRAYSCALE)
            combined_image[(sea_image > 10) & (combined_image != 2)] = 1
    
        # Check if sky labels exist and overlay with priority
        sky_image_path = os.path.join(sky_folder, image_name)
        if os.path.exists(sky_image_path):
            sky_image = cv2.imread(sky_image_path, cv2.IMREAD_GRAYSCALE)
            combined_image[(sky_image > 80) & (combined_image != 2)] = 3
        
        if color_mask==True:
            # Create a color image based on the combined labels
            colored_image = np.zeros((combined_image.shape[0], combined_image.shape[1], 3), dtype=np.uint8)
            for label, color in label_colors.items():
                mask = combined_image == label_priorities[label]
                colored_image[mask] = color
            
            if save==True:
                # Save the combined and colored image
                output_image_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_image_path, colored_image)
        else:
            if save==True:
                output_image_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_image_path, combined_image)
            else:
                combined_image = cv2.resize(combined_image, (SIZE_Y, SIZE_X))
                combined_images.append(combined_image)
    
    combined_images = np.array(combined_images)
    print("Combining complete.")
    return combined_images

def get_label(ship_folder, output_folder, label_color, color_mask=False, save=False, SIZE_X=256, SIZE_Y=256):
    """
    *This function gets masks of ships*
    Input:
        ship_folder: path of ship masks - text
        output_folder: output path wehere to save combined masks - text
        label_color: if you want to save masks colorful you assign colors - dict
        color_mask: if you want to save the masks colorful - Bool
        save:if you want to save the combined masks in output folder - Bool
        SIZE_X: width of images - int
        SIZE_Y: height of images - int
    Output:
        labeled_images: is masks of ships - numpy array
    """
    labeled_images = []
    
    # Loop through the images in the ship folder
    for image_name in os.listdir(ship_folder):
        image_path = os.path.join(ship_folder, image_name)
        ship_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
        # Initialize the combined image with ship labels
        labeled_image = np.zeros_like(ship_image, dtype=np.uint8)
    
        # Set ship labels to 2
        labeled_image[ship_image > 120] = 1
        
        if color_mask==True:
            # Create a color image based on the combined labels
            colored_image = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=np.uint8)
            for label, color in label_color.items():
                mask = labeled_image == 1
                colored_image[mask] = color
            
            if save==True:
                # Save the combined and colored image
                output_image_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_image_path, colored_image)
        else:
            if save==True:
                output_image_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_image_path, labeled_image)
            else:
                labeled_image = cv2.resize(labeled_image, (SIZE_Y, SIZE_X))
                labeled_images.append(labeled_image)
    
    labeled_images = np.array(labeled_images)
    print("Reading complete.")
    return labeled_images



def combine_labels_patches(ship_folder, sea_folder, sky_folder, output_folder, label_colors, label_priorities, color_mask=False, save=True, SIZE_X=256, SIZE_Y=256):
    """
    *This function combines all separate masks together*
    Input:
        ship_folder: path of ship masks - text
        sea_folder: path of sea masks - text
        sky_folder: path of sky masks - text
        output_folder: output path where to save combined masks - text
        label_colors: if you want to save masks colorful you assign colors - dict
        label_priorities: binary values of combined masks - dict
        color_mask: if you want to save the masks colorful - Bool
        save: if you want to save the combined masks in the output folder - Bool
        SIZE_X: width of images - int
        SIZE_Y: height of images - int
    Output:
        combined_images: combined masks - numpy array
    """
    combined_images = []
    
    ship_files, subfolder_names = get_image_paths(ship_folder)
    
    for index, image_path in enumerate(tqdm(ship_files)):
        image_name = os.path.basename(image_path)

        ship_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Initialize the combined image with ship labels
        combined_image = np.zeros_like(ship_image, dtype=np.uint8)

        # Set ship labels to 2
        combined_image[ship_image > 120] = 2

        # Check if sea labels exist and overlay with priority
        sea_image_path = os.path.join(sea_folder, subfolder_names[index], image_name)
        if os.path.exists(sea_image_path):
            sea_image = cv2.imread(sea_image_path, cv2.IMREAD_GRAYSCALE)
            combined_image[(sea_image > 10) & (combined_image != 2)] = 1

        # Check if sky labels exist and overlay with priority
        sky_image_path = os.path.join(sky_folder, subfolder_names[index], image_name)
        if os.path.exists(sky_image_path):
            sky_image = cv2.imread(sky_image_path, cv2.IMREAD_GRAYSCALE)
            combined_image[(sky_image > 80) & (combined_image != 2)] = 3

        if color_mask:
            # Create a color image based on the combined labels
            colored_image = np.zeros((combined_image.shape[0], combined_image.shape[1], 3), dtype=np.uint8)
            for label, color in label_colors.items():
                mask = combined_image == label_priorities[label]
                colored_image[mask] = color

            if save:
                # Save the combined and colored image
                output_image_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_image_path, colored_image)
        else:
            if save:
                output_image_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_image_path, combined_image)
            else:
                combined_images.append(combined_image)

    combined_images = np.array(combined_images)
    print("Combining complete.")
    return combined_images


def get_label_patches(ship_folder, output_folder, label_color, color_mask=False, save=True, SIZE_X=256, SIZE_Y=256):
    """
    *This function combines all separate masks together*
    Input:
        ship_folder: path of ship masks - text
        output_folder: output path where to save combined masks - text
        label_colors if you want to save masks colorful you assign colors - dict
        color_mask: if you want to save the masks colorful - Bool
        save: if you want to save the combined masks in the output folder - Bool
        SIZE_X: width of images - int
        SIZE_Y: height of images - int
    Output:
        labeled_images: masks of ships - numpy array
    """
    labeled_images = []
    
    ship_files, subfolder_names = get_image_paths(ship_folder)
    
    for index, image_path in enumerate(tqdm(ship_files)):
        image_name = os.path.basename(image_path)

        ship_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Initialize the combined image with ship labels
        labeled_image = np.zeros_like(ship_image, dtype=np.uint8)

        # Set ship labels to 2
        labeled_image[ship_image > 120] = 1

        if color_mask:
            # Create a color image based on the combined labels
            colored_image = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=np.uint8)
            for label, color in label_color.items():
                mask = labeled_image == 1
                colored_image[mask] = color

            if save:
                # Save the colored label image
                output_image_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_image_path, colored_image)
        else:
            if save:
                output_image_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_image_path, labeled_image)
            else:
                labeled_images.append(labeled_image)

    labeled_images = np.array(labeled_images)
    print("Combining complete.")
    return labeled_images    


if __name__ == '__main__':
    # Input folders containing labeled images
    sea_folder = r"C:/Users/novru/.spyder-py3/envs/Ship_Segmentation/Ship_dataset/COCO/COCO_new/sky_sea/sea/Train"
    sky_folder = r"C:/Users/novru/.spyder-py3/envs/Ship_Segmentation/Ship_dataset/COCO/COCO_new/sky_sea/sky/Train"
    ship_folder = r"C:/Users/novru/.spyder-py3/envs/Ship_Segmentation/Ship_dataset/COCO/COCO_new/Labeled_train_val2017/ship/Train"
    output_folder = r"C:/Users/novru/.spyder-py3/envs/Ship_Segmentation/Ship_dataset/COCO/COCO_new/All_Labeled"
    
    # Define label priorities
    label_priorities = {
        "ship": 2,
        "sea": 1,
        "sky": 3,
    }

    # Define RGB colors for labels
    label_colors = {
        "sea": (230, 7, 9),
        "sky": (230, 230, 6),
        "ship": (116, 244, 255),
    }
    
    creat_output_folder(output_folder)
    combine_labels(ship_folder, sea_folder, sky_folder, output_folder, label_colors, label_priorities, color_mask=False)
    