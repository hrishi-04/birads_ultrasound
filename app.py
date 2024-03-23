import streamlit as st
import pandas as pd
from PIL import Image
from PIL import Image, ImageChops
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import cv2
from tensorflow.keras.metrics import AUC
from keras.models import load_model
import os
import matplotlib.pyplot as plt



# Load the model
loaded_model = tf.keras.models.load_model('ultrasound_birad_model.hdf5', compile=False)

loaded_model.compile(optimizer=Adam(learning_rate=0.0007), loss='categorical_crossentropy', metrics=['accuracy'])

df = pd.read_csv('info.csv')

def cropBorders(img, l=0.01, r=0.01, u=0.04, d=0.04):

    nrows, ncols = img.shape

    # Get the start and end rows and columns
    l_crop = int(ncols * l)
    r_crop = int(ncols * (1 - r))
    u_crop = int(nrows * u)
    d_crop = int(nrows * (1 - d))

    cropped_img = img[u_crop:d_crop, l_crop:r_crop]

    return cropped_img

def minMaxNormalise(img):

    norm_img = img/255

    return norm_img

def globalBinarise(img, thresh, maxval):

    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval

    return binarised_img

def editMask(mask, ksize=(23, 23)):

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)

    edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Then dilate
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)

    return edited_mask

def sortContoursByArea(contours, reverse=True):

    # Sort contours based on contour area.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)

    # Construct the list of corresponding bounding boxes.
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]

    return sorted_contours, bounding_boxes

def xLargestBlobs(mask, top_x=None, reverse=True):

    # Find all contours from binarised image.
    # Note: parts of the image that you want to get should be white.
    contours, hierarchy = cv2.findContours(
        image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )

    n_contours = len(contours)

    # Only get largest blob if there is at least 1 contour.
    if n_contours > 0:

        # Make sure that the number of contours to keep is at most equal
        # to the number of contours present in the mask.
        if n_contours < top_x or top_x == None:
            top_x = n_contours

        # Sort contours based on contour area.
        sorted_contours, bounding_boxes = sortContoursByArea(
            contours=contours, reverse=reverse
        )

        # Get the top X largest contours.
        X_largest_contours = sorted_contours[0:top_x]

        # Create black canvas to draw contours on.
        to_draw_on = np.zeros(mask.shape, np.uint8)

        # Draw contours in X_largest_contours.
        X_largest_blobs = cv2.drawContours(
            image=to_draw_on,  # Draw the contours on `to_draw_on`.
            contours=X_largest_contours,  # List of contours to draw.
            contourIdx=-1,  # Draw all contours in `contours`.
            color=1,  # Draw the contours in white.
            thickness=-1,  # Thickness of the contour lines.
        )

    return n_contours, X_largest_blobs

def applyMask(img, mask):

    masked_img = img.copy()
    masked_img[mask == 0] = 0

    return masked_img

def checkLRFlip(mask):

    # Get number of rows and columns in the image.
    nrows, ncols = mask.shape
    x_center = ncols // 2
    y_center = nrows // 2

    # Sum down each column.
    col_sum = mask.sum(axis=0)
    # Sum across each row.
    row_sum = mask.sum(axis=1)

    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])

    if left_sum < right_sum:
        LR_flip = True
    else:
        LR_flip = False

    return LR_flip


def makeLRFlip(img):

    flipped_img = np.fliplr(img)

    return flipped_img

def clahe(img, clip=2.0, tile=(8, 8)):

    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    img_uint8 = img.astype("uint8")

    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img_uint8)

    return clahe_img

def pad(img):

    nrows, ncols = img.shape

    # If padding is required...
    if nrows != ncols:

        # Take the longer side as the target shape.
        if ncols < nrows:
            target_shape = (nrows, nrows)
        elif nrows < ncols:
            target_shape = (ncols, ncols)

        # pad.
        padded_img = np.zeros(shape=target_shape)
        padded_img[:nrows, :ncols] = img

    # If padding is not required...
    elif nrows == ncols:

        # Return original image.
        padded_img = img

    return padded_img

def preprocess_pipeline(img):

    image_crop = cropBorders(img, 0.01, 0.01, 0.04, 0.04)
    image_norm = minMaxNormalise(image_crop)
    mask_bin = globalBinarise(image_norm, 0.1, 1)
    mask_edit = editMask(mask_bin)
    n, mask_sort = xLargestBlobs(mask_edit, 1)
    image_process = applyMask(image_norm, mask_sort)
    image_clahe = clahe(image_process)
    image_clahe_process = applyMask(image_clahe, mask_sort)

    if(checkLRFlip(mask_edit)):
        image_clahe_process = makeLRFlip(image_clahe_process)

    image_pad = minMaxNormalise(pad(image_clahe_process))

    return image_pad, checkLRFlip(mask_edit)




def process_and_predict(folder_path):
    image_files = os.listdir(folder_path)
    results_df = pd.DataFrame(columns=['name', 'prediction', 'actual'])
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        image = image.convert('L')
        image_raw = np.array(image)
        crop2, _ = preprocess_pipeline(image_raw)
        crop2 = cv2.resize(crop2, (1000, 1000))
        crop2 = np.stack((crop2,) * 3, axis=-1)
        crop2 = np.expand_dims(crop2, axis=0)
        prediction = loaded_model.predict(crop2)
        class_label = np.argmax(prediction) + 2
        
        # Get the actual value from the DataFrame
        
        st.write("Image file ",image_file)
        str1 = str(image_file)
        str1 = str1.strip()
        st.write("string ",str1)
        actual = df.loc[df['Image_filename'].str.contains(str1, regex=False), 'BIRADS'].values[0]
        
        # Append the result to the DataFrame
        results_df = pd.concat([results_df, pd.DataFrame({'name': [image_file], 'prediction': [class_label], 'actual': [actual]})], ignore_index=True)
        
    return results_df

# Function to display images in a 3x3 grid
def display_images(folder_path, results_df, start):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.ravel()
    
    for i in range(start, start+9):
        if i < len(results_df):
            image_file = results_df.iloc[i]['name']
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)
            axs[i-start].imshow(image, cmap='gray')
            axs[i-start].set_title(f'Predicted: {results_df.iloc[i]["prediction"]}, Actual: {results_df.iloc[i]["actual"]}')
            axs[i-start].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

# Main part of the app
st.title('BIRADS Ultrasound App')

folder_path = st.text_input("Enter the folder path...", "")
if folder_path:
    if st.button('Process and Classify Images'):
        st.write("Processing and classifying images...")
        results_df = process_and_predict(folder_path)
        # Calculate the accuracy
        accuracy = (results_df['prediction'] == results_df['actual']).mean()
        accuracy_percent = format(accuracy * 100, '.2f')
        # Save results_df for later use
        results_df.to_csv('results.csv', index=False)

        # Initialize the accuracy_percent in the session state if it doesn't exist
        if 'accuracy_percent' not in st.session_state:
            st.session_state.accuracy_percent = accuracy_percent
        else:
            st.session_state.accuracy_percent = accuracy_percent

    # Load results_df from the saved file
    results_df = pd.read_csv('results.csv')
    # Retrieve the accuracy_percent from the session state
    st.write(f'The accuracy is: {st.session_state.accuracy_percent}%')
    # Display the images in pages
    page = st.number_input('Enter a page number', min_value=1, max_value=(len(results_df)+8)//9, value=1, step=1)
    start = (page-1)*9
    display_images(folder_path, results_df, start)