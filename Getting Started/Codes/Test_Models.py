#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:48:34 2023

@author: nmammadli
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from weighted_categorical_crossentropy import weighted_categorical_crossentropy
from keras.utils import normalize


def predict_image(model, img):
    # Expand dimensions to match the model's input shape
    input_image = np.expand_dims(img, axis=0)

    # Make predictions using your model

    predictions = model.predict(input_image)
    return predictions

def preprocess_image(image_path, color, SIZE_X, SIZE_Y):
    # Read the image based on the color argument (0 for grayscale, 1 for RGB)
    img = cv2.imread(image_path, color)

    if color == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to 256x256
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    
    if color == 0:
        # Expand the dimensions to make it compatible with a model expecting a single channel
        img = np.expand_dims(img, axis=-1)
    
    # Normalize the image
    img = normalize(img, axis=1)
    return img

def plot_multi_predictions(img, model_paths, label_cmap, th=0.99):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for i in range(4):
        for j in range(4):
            model_index = i * 4 + j
            if model_index < len(model_paths):
                # Load the model
                model_name=model_paths[model_index][:-4]
                model_name_1 = model_name.split('/')[-1]
                if 'weights' in model_paths[model_index]:
                    model = tf.keras.models.load_model(model_paths[model_index], custom_objects={'loss': weighted_categorical_crossentropy})
                else:
                    model = tf.keras.models.load_model(model_paths[model_index])
                    
                # Reset the RGB image for each new model
                predicted_image_rgb = np.zeros((img.shape[0], img.shape[1], 3))
                
                # Get predictions for the external image
                predictions = predict_image(model, img)
                predicted_img = np.argmax(predictions, axis=3)[0,:,:]
                
                # # Map labels to colors
                # for label in range(predictions.shape[-1]):
                #     label_mask = predictions[0, :, :, label]
                #     label_color = label_cmap(label / (predictions.shape[-1] - 1))
                #     label_mask[label_mask < th] = 0
                #     predicted_image_rgb += np.expand_dims(label_mask, axis=2) * label_color[:3]
                
                # Display the external image and predictions
                axes[i, j].imshow(predicted_img, cmap='jet')
                axes[i, j].axis('off')
                axes[i, j].set_title(f'Model {model_name_1}\nPredicted Image')

    plt.tight_layout()
    plt.show()

model_name1 = r"Models/Resized/Multi_UNet/Color/resized_MultiUNet_color/resized_MultiUNet_color.hdf5"
model_name2 = r"Models/Resized/Multi_UNet/Color/resized_MultiUNet_color_dropout/resized_MultiUNet_color_dropout.hdf5"
model_name3 = r"Models/Resized/Multi_UNet/Color/resized_MultiUNet_color_weights/resized_MultiUNet_color_weights.hdf5"
model_name4 = r"Models/Resized/Multi_UNet/Color/resized_MultiUNet_color_weights_dropout/resized_MultiUNet_color_weights_dropout.hdf5"

model_name5 = r"Models/Resized/Multi_UNet/Gray/resized_MultiUNet_gray/resized_MultiUNet_gray.hdf5"
model_name6 = r"Models/Resized/Multi_UNet/Gray/resized_MultiUNet_gray_dropout/resized_MultiUNet_gray_dropout.hdf5"
model_name7 = r"Models/Resized/Multi_UNet/Gray/resized_MultiUNet_gray_weights/resized_MultiUNet_gray_weights.hdf5"
model_name8 = r"Models/Resized/Multi_UNet/Gray/resized_MultiUNet_gray_weights_dropout/resized_MultiUNet_gray_weights_dropout.hdf5"

model_name9 = r"Models/Patched/Multi_UNet/Color/patched_MultiUNet_color/patched_MultiUNet_color.hdf5"
model_name10 = r"Models/Patched/Multi_UNet/Color/patched_MultiUNet_color_dropout/patched_MultiUNet_color_dropout.hdf5"
model_name11 = r"Models/Patched/Multi_UNet/Color/patched_MultiUNet_color_weights/patched_MultiUNet_color_weights.hdf5"
model_name12 = r"Models/Patched/Multi_UNet/Color/patched_MultiUNet_color_weights_dropout/patched_MultiUNet_color_weights_dropout.hdf5"

model_name13 = r"Models/Patched/Multi_UNet/Gray/patched_MultiUNet_gray/patched_MultiUNet_gray.hdf5"
model_name14 = r"Models/Patched/Multi_UNet/Gray/patched_MultiUNet_gray_dropout/patched_MultiUNet_gray_dropout.hdf5"
model_name15 = r"Models/Patched/Multi_UNet/Gray/patched_MultiUNet_gray_weights/patched_MultiUNet_gray_weights.hdf5"
model_name16 = r"Models/Patched/Multi_UNet/Gray/patched_MultiUNet_gray_weights_dropout/patched_MultiUNet_gray_weights_dropout.hdf5"

model_name17 = r"Models/Resized/Single_UNet/Color/resized_SingleUNet_color/resized_SingleUNet_color.hdf5"
model_name18 = r"Models/Resized/Single_UNet/Color/resized_SingleUNet_color_dropout/resized_SingleUNet_color_dropout.hdf5"
model_name19 = r"Models/Resized/Single_UNet/Color/resized_SingleUNet_color_weights/resized_SingleUNet_color_weights.hdf5"
model_name20 = r"Models/Resized/Single_UNet/Color/resized_SingleUNet_color_weights_dropout/resized_SingleUNet_color_weights_dropout.hdf5"

model_name21 = r"Models/Resized/Single_UNet/Gray/resized_SingleUNet_gray/resized_SingleUNet_gray.hdf5"
model_name22 = r"Models/Resized/Single_UNet/Gray/resized_SingleUNet_gray_dropout/resized_SingleUNet_gray_dropout.hdf5"
model_name23 = r"Models/Resized/Single_UNet/Gray/resized_SingleUNet_gray_weights/resized_SingleUNet_gray_weights.hdf5"
model_name24 = r"Models/Resized/Single_UNet/Gray/resized_SingleUNet_gray_weights_dropout/resized_SingleUNet_gray_weights_dropout.hdf5"

model_name25 = r"Models/Patched/Single_UNet/Color/patched_SingleUNet_color/patched_SingleUNet_color.hdf5"
model_name26 = r"Models/Patched/Single_UNet/Color/patched_SingleUNet_color_dropout/patched_SingleUNet_color_dropout.hdf5"
model_name27 = r"Models/Patched/Single_UNet/Color/patched_SingleUNet_color_weights/patched_SingleUNet_color_weights.hdf5"
model_name28 = r"Models/Patched/Single_UNet/Color/patched_SingleUNet_color_weights_dropout/patched_SingleUNet_color_weights_dropout.hdf5"

model_name29 = r"Models/Patched/Single_UNet/Gray/patched_SingleUNet_gray/patched_SingleUNet_gray.hdf5"
model_name30 = r"Models/Patched/Single_UNet/Gray/patched_SingleUNet_gray_dropout/patched_SingleUNet_gray_dropout.hdf5"
model_name31 = r"Models/Patched/Single_UNet/Gray/patched_SingleUNet_gray_weights/patched_SingleUNet_gray_weights.hdf5"
model_name32 = r"Models/Patched/Single_UNet/Gray/patched_SingleUNet_gray_weights_dropout/patched_SingleUNet_gray_weights_dropout.hdf5"


# Define the model paths
model_paths_color = [model_name1, model_name2, model_name3, model_name4,
              model_name9, model_name10, model_name11, model_name12,
              model_name17, model_name18, model_name19, model_name20,
              model_name25, model_name26, model_name27, model_name28]

model_paths_gray = [model_name5, model_name6, model_name7, model_name8,
              model_name13, model_name14, model_name15, model_name16,
              model_name21, model_name22, model_name23, model_name24,
              model_name29, model_name30, model_name31, model_name32]

SIZE_X = 256
SIZE_Y = 256
color0 = 0
color1 = 1
th = 0
label_cmap = plt.get_cmap('jet')
# Define the input image path
image_path1 = r"/unix/home/nmammadli/Downloads/Cruise-ships-1.png"
image_path2 = r"/unix/home/nmammadli/Downloads/pexels-matthew-barra-813011.png"
image_path3 = r"/unix/home/nmammadli/Downloads/f9nrycuxcaau-ds-woxv_cover.png"
image_path4 = r"/unix/home/nmammadli/Downloads/HERO_Viking-Octantis.png"

img1_gray = preprocess_image(image_path1, color0, SIZE_X, SIZE_Y)
img1_color = preprocess_image(image_path1, color1, SIZE_X, SIZE_Y)

img2_gray = preprocess_image(image_path2, color0, SIZE_X, SIZE_Y)
img2_color = preprocess_image(image_path2, color1, SIZE_X, SIZE_Y)

img3_gray = preprocess_image(image_path3, color0, SIZE_X, SIZE_Y)
img3_color = preprocess_image(image_path3, color1, SIZE_X, SIZE_Y)

img4_gray = preprocess_image(image_path4, color0, SIZE_X, SIZE_Y)
img4_color = preprocess_image(image_path4, color1, SIZE_X, SIZE_Y)

# Call the function to plot the model predictions for the first time
plot_multi_predictions(img1_gray, model_paths_gray, label_cmap, th)
# Call the function to plot the model predictions for the second time with the same image
plot_multi_predictions(img1_color, model_paths_color, label_cmap, th)

# Call the function to plot the model predictions for the first time
plot_multi_predictions(img2_gray, model_paths_gray, label_cmap, th)
# Call the function to plot the model predictions for the second time with the same image
plot_multi_predictions(img2_color, model_paths_color, label_cmap, th)

# Call the function to plot the model predictions for the first time
plot_multi_predictions(img3_gray, model_paths_gray, label_cmap, th)
# Call the function to plot the model predictions for the second time with the same image
plot_multi_predictions(img3_color, model_paths_color, label_cmap, th)

# Call the function to plot the model predictions for the first time
plot_multi_predictions(img4_gray, model_paths_gray, label_cmap, th)
# Call the function to plot the model predictions for the second time with the same image
plot_multi_predictions(img4_color, model_paths_color, label_cmap, th)