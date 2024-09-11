# Import necessary libraries
import os
import numpy as np
import pandas as pd
from PIL import Image  # For image processing
import tables  # For HDF5 file handling
import concurrent.futures  # For parallel processing (multithreading)

# Function to read an image, resize it to 224x224, and convert it to a numpy array of uint8 type
def read_and_convert_image(img_path):
    img = Image.open(img_path)  # Open image
    img = np.array(img.resize((224, 224)))  # Resize to 224x224
    img = img.astype(np.uint8)  # Convert to uint8 type
    return img

# Function to save image patch names to a CSV file
def save_patches_to_csv(image_list, csv_path):
    # Flatten the list of image paths
    image_paths = [item for sublist in image_list for item in sublist]
    
    # Extract just the file names (without extensions) from the image paths
    patches_name = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]
    
    # Create a pandas DataFrame with the patch names
    patient_df = pd.DataFrame(patches_name, columns=['Patch_Name'])
    
    # Save DataFrame to CSV
    patient_df.to_csv(csv_path, index=False)  # Add index=False to avoid unnecessary index column

# Function to process images and store them in HDF5 format
def make_hdf52(H5D_Train_path, Image_path, WES_file_list, max_workers=8):
    # Open or create an HDF5 file to store the patches
    with tables.open_file(H5D_Train_path + ".hdf5", mode='w') as store_train:
        
        # Define the datatype for the images (uint8 for 8-bit images) and initial data shape
        img_dtype = tables.UInt8Atom()  # Atom is used for defining the datatype
        data_shape = (0, 224, 224, 3)  # Initial empty shape (0 rows, with images of size 224x224x3)
        
        # Create an extendable array to store image patches
        storage_train = store_train.create_earray(
            store_train.root,  # Root group in HDF5 file
            atom=img_dtype,  # Data type of the array
            name='patches',  # Dataset name
            shape=data_shape  # Initial shape
        )
        
        # Initialize a list to store the paths of processed training images
        Train_paths = []
        
        # Use ThreadPoolExecutor for parallel processing of images
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Iterate through the list of training filenames
            for index, filename in enumerate(WES_file_list):
                print(f"Processing train image {index+1}/{len(WES_file_list)}")

                # Get all image file paths inside the given directory
                image_dir = os.path.join(Image_path, filename)  # Directory for current sample
                image_paths_Train = [
                    os.path.join(image_dir, img) 
                    for img in sorted(os.listdir(image_dir))
                ]

                # Use multithreading to read and process images in parallel
                images = list(executor.map(read_and_convert_image, image_paths_Train))
                
                # Append the processed images to the HDF5 dataset
                storage_train.append(np.array(images))

                # Store image paths for CSV export
                Train_paths.append(image_paths_Train)
        
    # Save the names of patches to a CSV file
    save_patches_to_csv(Train_paths, H5D_Train_path + ".csv")

# Main function to execute the script
if __name__ == '__main__':
    Image_path = ''  # Path to the folder containing images
    H5D_Train_path = ''  # Path to store the HDF5 and CSV files
    
    # Read a CSV file to get the list of sample IDs
    BZ_Train = pd.read_csv('', encoding='GBK')
    
    # Generate a list of filenames with the '-1' suffix (assuming this is the required format)
    Filename_Train = [name + '-1' for name in BZ_Train['SampleID'].values]
    
    # Call the function to create the HDF5 file and process the images
    make_hdf52(H5D_Train_path, Image_path, Filename_Train)
