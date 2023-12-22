#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 23:32:44 2023

@author: nmammadli
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import normalize
from Codes.weighted_categorical_crossentropy import weighted_categorical_crossentropy


def predict_image(model, img):
    # Expand dimensions to match the model's input shape
    input_image = np.expand_dims(img, axis=0)

    # Make predictions using your model

    predictions = model.predict(input_image)
    return predictions

def preprocess_image(image_path, SIZE_X, SIZE_Y):
    # Read the image based on the color argument (0 for grayscale, 1 for RGB)
    img_color = cv2.imread(image_path, 1)
    img_gray = cv2.imread(image_path, 0)

    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    
    # Resize the image to 256x256
    img_color = cv2.resize(img_color, (SIZE_Y, SIZE_X))
    img_gray = cv2.resize(img_gray, (SIZE_Y, SIZE_X))
    
    # Expand the dimensions to make it compatible with a model expecting a single channel
    img_gray = np.expand_dims(img_gray, axis=-1)
    
    # Normalize the image
    img_color = normalize(img_color, axis=1)
    img_gray = normalize(img_gray, axis=1)
    return img_gray, img_color

def plot_multi_predictions(image_path, model_paths_gray, model_paths_color, SIZE_X=256, SIZE_Y=256):
    fig, axes = plt.subplots(2, 3, figsize=(16, 16))
    
    # Preprocess the image
    img_gray, img_color = preprocess_image(image_path, SIZE_X, SIZE_Y)
    
    # Process grayscale images
    for i in range(3):
        model_index = i
        if model_index < len(model_paths_gray):
            model_name = model_paths_gray[model_index][:-4]
            model_name_1 = model_name.split('/')[-1]
            if 'weights' in model_paths_gray[model_index]:
                model = tf.keras.models.load_model(model_paths_gray[model_index], custom_objects={'loss': weighted_categorical_crossentropy})
            else:
                model = tf.keras.models.load_model(model_paths_gray[model_index])
            
            
            # Get predictions for the grayscale image
            predictions = predict_image(model, img_gray)
            predicted_img = np.argmax(predictions, axis=3)[0, :, :]
            
            # Display the grayscale image and predictions
            axes[0, i].imshow(predicted_img, cmap='jet')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Model {model_name_1}\nPredicted Image (Gray)')

    # Process colored images
    for i in range(3):
        model_index = i
        if model_index < len(model_paths_color):
            model_name = model_paths_color[model_index][:-4]
            model_name_1 = model_name.split('/')[-1]
            if 'weights' in model_paths_color[model_index]:
                model = tf.keras.models.load_model(model_paths_color[model_index], custom_objects={'loss': weighted_categorical_crossentropy})
            else:
                model = tf.keras.models.load_model(model_paths_color[model_index])
            
            # Get predictions for the colored image
            predictions = predict_image(model, img_color)
            predicted_img = np.argmax(predictions, axis=3)[0, :, :]
            
            # Display the colored image and predictions
            axes[1, i].imshow(predicted_img, cmap='jet')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'Model {model_name_1}\nPredicted Image (Colored)')

    plt.tight_layout()
    plt.show()
    
    
def plot_predictions(image_path, model_paths, SIZE_X=256, SIZE_Y=256):
    # Preprocess the image
    img_gray, img_color = preprocess_image(image_path, SIZE_X, SIZE_Y)
    
    for model_path in model_paths:
        model_name = model_path[:-4].split('/')[-1]

        if 'weights' in model_path:
            model = tf.keras.models.load_model(model_path, custom_objects={'loss': weighted_categorical_crossentropy})
        else:
            model = tf.keras.models.load_model(model_path)

        # Choose the appropriate image for prediction
        img = img_gray if 'gray' in model_name.lower() else img_color

        # Get predictions for the image
        predictions = predict_image(model, img)
        predicted_img = np.argmax(predictions, axis=3)[0, :, :]

        # Display the image and predictions
        plt.figure(figsize=(6, 6))
        plt.imshow(predicted_img, cmap='jet')
        plt.axis('off')
        plt.title(f'Model {model_name}\nPredicted Image')
        plt.show()
        
def plot_best_prediction(image_path, model_path, SIZE_X=256, SIZE_Y=256):
    # Preprocess the image
    img_gray, img_color = preprocess_image(image_path, SIZE_X, SIZE_Y)
    
    model_name = model_path[:-4].split('/')[-1]

    if 'weights' in model_path:
        model = tf.keras.models.load_model(model_path, custom_objects={'loss': weighted_categorical_crossentropy})
    else:
        model = tf.keras.models.load_model(model_path)

    # Choose the appropriate image for prediction
    img = img_gray if 'gray' in model_name.lower() else img_color

    # Get predictions for the image
    predictions = predict_image(model, img)
    predicted_img = np.argmax(predictions, axis=3)[0, :, :]

    # Display the image and predictions
    plt.figure(figsize=(6, 6))
    plt.imshow(predicted_img, cmap='jet')
    plt.axis('off')
    plt.title(f'Model {model_name}\nPredicted Image')
    plt.show()        
    


def process_images(input_folder, model_path, SIZE_X=256, SIZE_Y=256, plot=False):
    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Load the model
    model_name = model_path[:-5].split('/')[-1]
    if 'weights' in model_path:
        model = tf.keras.models.load_model(model_path, custom_objects={'loss': weighted_categorical_crossentropy})
    else:
        model = tf.keras.models.load_model(model_path)

    # Create 'Results' folder if it doesn't exist
    results_folder = os.path.join(input_folder, f'Results_of_{model_name}')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Process each image in the folder
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)

        # Preprocess the image
        img_gray, img_color = preprocess_image(image_path, SIZE_X, SIZE_Y)

        # Preprocess the image
        img = img_gray if 'gray' in model_name.lower() else img_color
        
        # Get predictions for images
        predictions = predict_image(model, img)
        
        predicted_img = np.argmax(predictions, axis=3)[0, :, :]
        
        if plot==True:
            fig, axes = plt.subplots(1, 2, figsize=(16, 16))
            # Display the image and predictions
            img_color = cv2.imread(image_path, 1)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            
            axes[0].imshow(img_color)
            axes[0].axis('off')
            
            axes[1].imshow(predicted_img, cmap='jet')
            axes[1].axis('off')
            
            plt.figure(figsize=(6, 6))
            plt.imshow(predicted_img, cmap='jet')
            plt.axis('off')
            plt.title(f'Model {model_name}\nPredicted Image')
            plt.show()        

        # Save the segmented image to the 'Results' folder
        result_image_path = os.path.join(results_folder, f"{image_file.split('.')[0]}_segmented.png")
        plt.imsave(result_image_path, predicted_img, cmap='jet')