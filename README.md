# Semantic-Segmentation-of-Watercrafts-using-UNet-Model

Abstract - Semantic segmentation plays a crucial role in computer vision applications, enabling the differentiation of objects within an image by machines. The focal point of this project is the task of semantic segmentation of ships in maritime scenes. The primary objective is the development of an accurate and efficient model using the UNet architecture. The MariBoats dataset, consisting of images instance-segmented with various watercraft, was employed for this purpose. Furthermore, the incorporation of semantic scene understanding involved the utilization of a pre-trained model based on the ADE20K dataset for the labeling of sea and sky regions. The project encompassed approaches utilizing both unpatched and patchified images for ship segmentation, as well as the joint segmentation of ships, sea, and sky.

## Functions in Getting Started

### 1. `plot_multi_predictions`

This function is used to plot predictions from 6 models in a single plot. It creates a 2x3 subplot to display the predictions. The function takes the following inputs:
  image_path: Path to the input image.
  model_paths_gray: List of paths to models for grayscale images.
  model_paths_color: List of paths to models for colored images.

### 2. `plot_best_prediction`

This function is used to plot predictions from a single model. It takes the following inputs:
  image_path: Path to the input image.
  model_path: Path to the model for prediction.

### 3. `plot_predictions`

This function is used to plot predictions from multiple models separately. It takes the following inputs:
  image_path: Path to the input image.
  model_paths: List of paths to all models for prediction.
### 4. `process_images`

This function is used to segment all images in a given folder and save the results in a 'Result' folder within the input folder. It also has an option to plot the segmented images if the plot argument is set to True. It takes the following inputs:
  input_folder: Path to the folder containing input images.
  model_path: Path to the model for segmentation.
  plot: Boolean argument (True or False) to control whether to plot the segmented images.
