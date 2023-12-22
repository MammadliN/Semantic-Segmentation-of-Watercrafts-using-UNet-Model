# Semantic-Segmentation-of-Watercrafts-using-UNet-Model

Abstract - Semantic segmentation plays a crucial role in computer vision applications, enabling the differentiation of objects within an image by machines. The focal point of this project is the task of semantic segmentation of ships in maritime scenes. The primary objective is the development of an accurate and efficient model using the UNet architecture. The MariBoats dataset, consisting of images instance-segmented with various watercraft, was employed for this purpose. Furthermore, the incorporation of semantic scene understanding involved the utilization of a pre-trained model based on the ADE20K dataset for the labeling of sea and sky regions. The project encompassed approaches utilizing both unpatched and patchified images for ship segmentation, as well as the joint segmentation of ships, sea, and sky.


## Functions in Getting Started/main.py

### 1. `plot_multi_predictions`

This function is used to plot predictions from 6 models in a single plot. It creates a 2x3 subplot to display the predictions. The function takes the following inputs:
  * `image_path`: Path to the input image.
  * `model_paths_gray`: List of paths to models for grayscale images.
  * `model_paths_color`: List of paths to models for colored images.
  * `SIZE_X`: Size of images along the x-axis (set to 256 for this project).
  * `SIZE_Y`: Size of images along the y-axis (set to 256 for this project).

#### Usage
```python
from Func import plot_multi_predictions

#Models
model_name1 = r"Models/resized_MultiUNet_color_dropout.hdf5"
model_name2 = r"Models/resized_MultiUNet_gray_dropout.hdf5"
model_name3 = r"Models/resized_SingleUNet_color_dropout.hdf5"
model_name4 = r"Models/resized_SingleUNet_gray_dropout.hdf5"
model_name5 = r"Models/resized_SingleUNet_color_weights.hdf5"
model_name6 = r"Models/resized_SingleUNet_gray_weights.hdf5"

# Define the models lists
model_paths_color = [model_name1, model_name3, model_name5]
model_paths_gray = [model_name2, model_name4, model_name6]

# Define the input image path
image_path1 = r"Test Images/Cruise-ships-1.png"

#Plotting
plot_multi_predictions(image_path1, model_paths_gray, model_paths_color)
```

### 2. `plot_best_prediction`

This function is used to plot predictions from a single model. It takes the following inputs:
  * `image_path`: Path to the input image.
  * `model_path`: Path to the model for prediction.
  * `SIZE_X`: Size of images along the x-axis (set to 256 for this project).
  * `SIZE_Y`: Size of images along the y-axis (set to 256 for this project).

#### Usage
```python
from Func import plot_best_prediction

#Models
model_name1 = r"Models/resized_MultiUNet_color_dropout.hdf5"
model_name4 = r"Models/resized_SingleUNet_gray_dropout.hdf5"

# Define the input image path
image_path1 = r"Test Images/Cruise-ships-1.png"

#Plotting
plot_best_prediction(image_path1, model_name1) # Color
plot_best_prediction(image_path1, model_name4) # Gray
```

### 3. `plot_predictions`

This function is used to plot predictions from multiple models separately. It takes the following inputs:
  * `image_path`: Path to the input image.
  * `model_path`: List of paths to all models for prediction.
  * `SIZE_X`: Size of images along the x-axis (set to 256 for this project).
  * `SIZE_Y`: Size of images along the y-axis (set to 256 for this project).

#### Usage
```python
from Func import plot_predictions

#Models
model_name1 = r"Models/resized_MultiUNet_color_dropout.hdf5"
model_name2 = r"Models/resized_MultiUNet_gray_dropout.hdf5"
model_name3 = r"Models/resized_SingleUNet_color_dropout.hdf5"
model_name4 = r"Models/resized_SingleUNet_gray_dropout.hdf5"
model_name5 = r"Models/resized_SingleUNet_color_weights.hdf5"
model_name6 = r"Models/resized_SingleUNet_gray_weights.hdf5"

# Define the models lists
model_paths_bests = [model_name1, model_name2, model_name3, model_name4, model_name5, model_name6]

# Define the input image path
image_path1 = r"Test Images/Cruise-ships-1.png"

#Plotting
plot_predictions(image_path1, model_paths_bests)
```

### 4. `process_images`

This function is used to segment all images in a given folder and save the results in a 'Result_of_{`model_name`}' folder within the input folder. It also has an option to plot the segmented images if the plot argument is set to True. It takes the following inputs:
  * `input_folder`: Path to the folder containing input images.
  * `model_path`: Path to the model for segmentation.
  * `SIZE_X`: Size of images along the x-axis (set to 256 for this project).
  * `SIZE_Y`: Size of images along the y-axis (set to 256 for this project).
  * `plot`: Boolean argument (True or False) to control whether to plot the segmented images.

#### Usage
```python
from Func import process_images

#Models
model_name1 = r"Models/resized_MultiUNet_color_dropout.hdf5"

# Define the input folder path
input_folder = r"Ship_dataset/COCO/train2017"

#Plotting
process_images(input_folder, model_name1)
```
