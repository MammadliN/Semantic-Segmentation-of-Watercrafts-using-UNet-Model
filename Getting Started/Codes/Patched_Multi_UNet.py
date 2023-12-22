# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 02:04:13 2023

@author: Novruz Mammadli
"""

from UNet_Model import multi_unet_model #softmax 

from Combine_Labels import combine_labels_patches, get_image_paths

from weighted_categorical_crossentropy import weighted_categorical_crossentropy

from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.utils import class_weight
from keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import time


def read_train_img(img_path, mask_path, SIZE_Y, SIZE_X, color=0, ending="*.jpg", mask=False):
    #Capture training image info as a list
    train_images = []
    image_paths, subfolder_names = get_image_paths(img_path)
    for directory_path in image_paths:
        for img_path in glob.glob(os.path.join(directory_path)):
            img = cv2.imread(img_path, color)
            if color == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (SIZE_Y, SIZE_X))
            train_images.append(img)
           
    #Convert list to array for machine learning processing        
    train_images = np.array(train_images)
    if color==0:
        train_images = np.expand_dims(train_images, axis=-1)
    
    train_masks = [] 
    if mask==True:
        #Capture mask/label info as a list
        for directory_path in glob.glob(mask_path):
            for mask_path in glob.glob(os.path.join(directory_path, ending)):
                mask = cv2.imread(mask_path, color)
                mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
                train_masks.append(mask)
                
        #Convert list to array for machine learning processing          
        train_masks = np.array(train_masks)
        np.unique(train_masks)
    return train_images, train_masks

###############################################

def encode_labels(train_masks):
    #Encode labels... but multi dim array so need to flatten, encode and reshape
    labelencoder = LabelEncoder()
    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
    
    np.unique(train_masks_encoded_original_shape)
    return train_masks_encoded_original_shape, train_masks_reshaped_encoded

#################################################

def prepare_data_for_segmentation(train_images, train_masks_encoded_original_shape, n_classes):
    train_images1=train_images
    train_images = normalize(train_images, axis=1)

    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

    #Create a subset of data for quick testing
    #Picking 10% for testing and remaining for training
    X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)
    _, X_test_unnormalized, _, _ = train_test_split(train_images1, train_masks_input, test_size = 0.10, random_state = 0)
    
    #Further split training data t a smaller subset for quick testing of models
    X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)
    
    print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 


    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
    
    
    
    test_masks_cat = to_categorical(y_test, num_classes=n_classes)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))
    
    return train_images, X_test, X_train, y_train_cat, y_test_cat, y_test, X_test_unnormalized

###############################################################

def compute_weight(train_masks_reshaped_encoded, class_weights='balanced'):
    class_weights = class_weight.compute_class_weight(class_weight=class_weights,
                                                      classes = np.unique(train_masks_reshaped_encoded),
                                                      y = train_masks_reshaped_encoded)
    # class_weights = {i : class_weights[i] for i in range(n_classes)}
    # class_weights = class_weights.tolist()
    print("Class weights are...:", class_weights)
    return class_weights

def get_model(X_train, X_test, y_train_cat, y_test_cat, n_classes, dropout, weights, class_weights,
              optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
              batch_size = 32, verbose=1, epochs=50, model_name='UNet_model.hdf5'):
    
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    
    model = multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, dropout=dropout)
    
    # model.summary()
    
    #If starting with pre-trained weights.  
    #model.load_weights('???.hdf5')
    if weights==False:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train_cat,
                            batch_size = batch_size, 
                            verbose=1, 
                            epochs=epochs, 
                            validation_data=(X_test, y_test_cat), 
                            #class_weight=class_weights,
                            shuffle=False)
    elif weights==True:
        # weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(class_weights)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        history = model.fit(X_train, y_train_cat,
                            batch_size = batch_size, 
                            verbose=1, 
                            epochs=epochs, 
                            validation_data=(X_test, y_test_cat), 
                            # class_weight=class_weights,
                            shuffle=False)
                        
    
    
    model.save(model_name)
    ############################################################
    
    #Evaluate the model
    	# evaluate model
    _, acc = model.evaluate(X_test, y_test_cat)
    print("Accuracy is = ", (acc * 100.0), "%")
    
    return model, history

def train_and_evaluate_models(model_names, train_images, train_masks, X_train, X_test, X_test_unnormalized, y_train_cat, y_test_cat, y_test, test_img_number, n_classes, class_weights, verbose=1, epochs=50):
    models = []
    historys = []
    times = []
    for i, model_name in enumerate(model_names):
        start_time = time.time()
        model_name_1 = model_name.split('/')[-1]
        # Check for 'dropout' and 'weights' in the model_name
        if 'dropout' in model_name_1 and 'weights' in model_name_1:
            print(f"training of '{model_name_1}' model started.")
            dropout = True
            weights = True
            model, history = get_model(X_train, X_test, y_train_cat, y_test_cat, n_classes, dropout, weights, class_weights,
                                       optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
                                       batch_size=16, verbose=verbose, epochs=epochs, model_name=model_name)
        elif 'dropout' in model_name_1:
            print(f"training of '{model_name_1}' model started.")
            dropout = True
            weights = False
            model, history = get_model(X_train, X_test, y_train_cat, y_test_cat, n_classes, dropout, weights, class_weights,
                                       optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
                                       batch_size=16, verbose=verbose, epochs=epochs, model_name=model_name)
        elif 'weights' in model_name_1:
            print(f"training of '{model_name_1}' model started.")
            dropout = False
            weights = True
            model, history = get_model(X_train, X_test, y_train_cat, y_test_cat, n_classes, dropout, weights, class_weights,
                                       optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
                                       batch_size=16, verbose=verbose, epochs=epochs, model_name=model_name)
        elif 'dropout' not in model_name_1 or 'weights' not in model_name_1:
            print(f"training of '{model_name_1}' model started.")
            dropout = False
            weights = False
            model, history = get_model(X_train, X_test, y_train_cat, y_test_cat, n_classes, dropout, weights, class_weights,
                                       optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
                                       batch_size=16, verbose=verbose, epochs=epochs, model_name=model_name)
        else:
            print(f"'name of {model_name_1}' model has problem. Check the name again")
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        plot_acc_loss(history)
        IOU(train_images, train_masks, model, X_test, y_test, n_classes)
        plot_random(model, model_name_1, X_test_unnormalized, X_test, y_test, test_img_number)

        print(f"Elapsed time of {model_name_1} model: {elapsed_time} seconds")
        times.append(elapsed_time)
        models.append(model)
        historys.append(history)
        
    plot_results(models, model_names, X_test_unnormalized, X_test, y_test, test_img_number)
        
    return models, historys, times

##################################

def plot_acc_loss(history):
    #plot the training and validation accuracy and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    plt.plot(epochs, acc, 'y', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

##################################  

def IOU(train_images, train_masks, model, X_test, y_test, n_classes):
    #IOU
    y_pred=model.predict(X_test)
    y_pred_argmax=np.argmax(y_pred, axis=3)
        
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())
    
    #To calculate I0U for each class...
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)
    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
    class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
    class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])
    
    print("IoU for class1 is: ", class1_IoU)
    print("IoU for class2 is: ", class2_IoU)
    print("IoU for class3 is: ", class3_IoU)
    print("IoU for class4 is: ", class4_IoU)
    
    plt.imshow(train_images[0, :,:,0], cmap='gray')
    plt.imshow(train_masks[0], cmap='gray')
    
#######################################################################

def plot_random(model, model_name, X_test_unnormalized, X_test, y_test, test_img_number):
    #Predict on a few images
    test_img_unnormalized = X_test_unnormalized[test_img_number]
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = model.predict(test_img_input)
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    model_name = model_name.split('/')[-1]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img_unnormalized)
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    # plt.colorbar()
    plt.subplot(233)
    plt.title(f'Prediction of: {model_name}')
    plt.imshow(predicted_img, cmap='jet')
    # plt.colorbar()
    plt.show()
    
def plot_results(models, model_names, X_test_unnormalized, X_test, y_test, test_img_number):
    test_img_unnormalized = X_test_unnormalized[test_img_number]
    ground_truth = y_test[test_img_number]
    
    # Create a subplot for each model
    num_models = len(models)
    plt.figure(figsize=(16, 12))
    
    # Display the unnormalized test image and ground truth in the first row
    plt.subplot(2, num_models + 1, 1)
    plt.title('Testing Image')
    plt.imshow(test_img_unnormalized)
    
    plt.subplot(2, num_models + 1, 2)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:, :, 0], cmap='jet')
    
    # Display the predictions for each model in the second row
    for i in range(num_models):
        model = models[i]
        model_name = model_names[i]
        model_name = model_name.split('/')[-1]
        test_img_input = np.expand_dims(X_test[test_img_number], 0)
        prediction = model.predict(test_img_input)
        predicted_img = np.argmax(prediction, axis=3)[0, :, :]
        
        plt.subplot(2, num_models + 1, num_models + 2 + i)
        plt.title(f'Prediction of: {model_name}')
        plt.imshow(predicted_img, cmap='jet')
    
    plt.show()

#####################################################################


if __name__ == '__main__':
    
    start_time = time.time()
    
    # Input folders containing labeled images
    patched_sea_folder = r"/home/nmammadli/Downloads/patches/sea_patches"
    patched_sky_folder = r"/home/nmammadli/Downloads/patches/sky_patches"
    patched_ship_folder = r"/home/nmammadli/Downloads/patches/ship_patches"
    patched_output_folder = r"/home/nmammadli/Downloads/patches/result"
    
    # Color & Patched Model names
    model_name9 = r"Models/Patched/Multi_UNet/Color/patched_MultiUNet_color.hdf5"
    model_name10 = r"Models/Patched/Multi_UNet/Color/patched_MultiUNet_color_dropout.hdf5"
    model_name11 = r"Models/Patched/Multi_UNet/Color/patched_MultiUNet_color_weights.hdf5"
    model_name12 = r"Models/Patched/Multi_UNet/Color/patched_MultiUNet_color_weights_dropout.hdf5"
    
    # Gray & Patched Model names
    model_name13 = r"Models/Patched/Multi_UNet/Gray/patched_MultiUNet_gray.hdf5"
    model_name14 = r"Models/Patched/Multi_UNet/Gray/patched_MultiUNet_gray_dropout.hdf5"
    model_name15 = r"Models/Patched/Multi_UNet/Gray/patched_MultiUNet_gray_weights.hdf5"
    model_name16 = r"Models/Patched/Multi_UNet/Gray/patched_MultiUNet_gray_weights_dropout.hdf5"
    
    model_names3 = [model_name9, model_name10, model_name11, model_name12]
    model_names4 = [model_name13, model_name14, model_name15, model_name16]
    
    #Resizing images, if needed
    test_img_number = 23 #5, 10
    epochs = 50
    SIZE_X = 256 
    SIZE_Y = 256
    n_classes = 4 #Number of classes for segmentation
    color0 = 0
    color1 = 1

    # Define label priorities
    label_priorities = {"ship": 2, "sea": 1, "sky": 3,}

    # Define RGB colors for labels
    label_colors = {"sea": (230, 7, 9), "sky": (230, 230, 6), "ship": (116, 244, 255),}

    train_masks_patched = combine_labels_patches(patched_ship_folder, patched_sea_folder, patched_sky_folder, patched_output_folder, label_colors, label_priorities, color_mask=False, save=False)
    
    img_path = r"/home/nmammadli/Downloads/patches/images_patches"
    mask_path = r"/home/nmammadli/Downloads/patches/images_patches"
    
    # Resized Color & Gray
    train_images_color, train_masks_color_not_use = read_train_img(img_path, mask_path, SIZE_Y, SIZE_X, color=color1, ending="*.jpg")
    train_images_gray, train_masks_gray_not_use = read_train_img(img_path, mask_path, SIZE_Y, SIZE_X, color=color0, ending="*.jpg")
     
    train_masks_encoded_original_shape, train_masks_reshaped_encoded = encode_labels(train_masks_patched)
    
    train_images_color, X_test_color, X_train_color, y_train_cat_color, y_test_cat_color, y_test_color, X_test_unnormalized_color = prepare_data_for_segmentation(train_images_color, train_masks_encoded_original_shape, n_classes)
    train_images_gray, X_test_gray, X_train_gray, y_train_cat_gray, y_test_cat_gray, y_test_gray, X_test_unnormalized_gray = prepare_data_for_segmentation(train_images_gray, train_masks_encoded_original_shape, n_classes)
    class_weights = compute_weight(train_masks_reshaped_encoded, class_weights='balanced')
    
    # test_img_number = random.randint(0, len(X_test_unnormalized_color))
    models_color, historyies_color, times_color = train_and_evaluate_models(model_names3, train_images_color, train_masks_patched, X_train_color, X_test_color, X_test_unnormalized_color, y_train_cat_color, y_test_cat_color, y_test_color, test_img_number, n_classes, class_weights, epochs=epochs)
    models_gray, historyies_gray, times_gray = train_and_evaluate_models(model_names4, train_images_gray, train_masks_patched, X_train_gray, X_test_gray, X_test_unnormalized_gray, y_train_cat_gray, y_test_cat_gray, y_test_gray, test_img_number, n_classes, class_weights, epochs=epochs)