#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 23:32:44 2023

@author: nmammadli
"""

from Func import plot_multi_predictions, plot_best_prediction, plot_predictions, process_images


if __name__ == '__main__':
    #Models
    model_name1 = r"Models/resized_MultiUNet_color_dropout.hdf5"
    model_name2 = r"Models/resized_MultiUNet_gray_dropout.hdf5"
    model_name3 = r"Models/resized_SingleUNet_color_dropout.hdf5"
    model_name4 = r"Models/resized_SingleUNet_gray_dropout.hdf5"
    model_name5 = r"Models/resized_SingleUNet_color_weights.hdf5"
    model_name6 = r"Models/resized_SingleUNet_gray_weights.hdf5"
    
    # Define the model paths
    model_paths_color = [model_name1, model_name3, model_name5]
    model_paths_gray = [model_name2, model_name4, model_name6]
    model_paths_bests = [model_name1, model_name2, model_name3, model_name4, model_name5, model_name6]
    
    SIZE_X = 256
    SIZE_Y = 256
    
    # Define the input image path
    input_folder = r"Ship_dataset/COCO/train2017"
    image_path1 = r"Test Images/Cruise-ships-1.png"
    image_path2 = r"Test Images/pexels-matthew-barra-813011.png"
    image_path3 = r"Test Images/f9nrycuxcaau-ds-woxv_cover.png"
    image_path4 = r"Test Images/HERO_Viking-Octantis.png"
    
    
    #Plotting
    # plot_multi_predictions(image_path1, model_paths_gray, model_paths_color)
    # plot_best_prediction(image_path1, model_name1)
    # plot_best_prediction(image_path1, model_name4)
    # plot_predictions(image_path1, model_paths_bests)
    
    # plot_multi_predictions(image_path2, model_paths_gray, model_paths_color)
    # plot_best_prediction(image_path2, model_name1)
    # plot_best_prediction(image_path2, model_name4)
    # plot_predictions(image_path2, model_paths_bests)
    
    # plot_multi_predictions(image_path3, model_paths_gray, model_paths_color)
    # plot_best_prediction(image_path3, model_name1)
    # plot_best_prediction(image_path3, model_name4)
    # plot_predictions(image_path3, model_paths_bests)
    
    # plot_multi_predictions(image_path4, model_paths_gray, model_paths_color)
    # plot_best_prediction(image_path4, model_name1)
    # plot_best_prediction(image_path4, model_name4)
    # plot_predictions(image_path4, model_paths_bests)
    
    # process_images(input_folder, model_name1)