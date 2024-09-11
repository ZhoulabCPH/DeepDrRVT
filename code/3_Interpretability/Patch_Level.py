import tables
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy import stats
import os
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Load Dataset and Merge Patch Data with Clinical Data
def Dataset(H5D_File, H5DImagePath, PreRVTS):
    """
    Load the patch-level data and clinical data from an HDF5 file and merge based on patient names.

    Parameters:
    H5D_File (DataFrame): DataFrame containing patch-level data.
    H5DImagePath (str): Path to the HDF5 file containing patch images.
    PreRVTS (DataFrame): DataFrame containing clinical data such as RVT and PreRVT.

    Returns:
    DataFrame: Merged dataset of patch-level and clinical data.
    HDF5 File: HDF5 dataset containing patch images.
    """
    store = tables.open_file(H5DImagePath, mode='r')
    
    # Process the patch-level data
    H5D_File['Name'] = [name.split("_")[0] + '_' + name.split("_")[-1] for name in H5D_File['Patch_Name'].values]
    Mydatasets = H5D_File
    Mydatasets['Patient_Name'] = [name.split("_")[0] for name in Mydatasets['Name'].values]
    Mydatasets['Coordinates'] = [name.split("_")[1] for name in Mydatasets['Name'].values]
    Mydatasets['Index_Image'] = np.arange(0, Mydatasets.shape[0])

    patch_images = store.root.patches

    # Process clinical data
    PreRVTS.columns = ['Patient_Name', 'RVT', 'PreRVT']
    PreRVTS['Patient_Name'] = np.array(PreRVTS['Patient_Name'].values, dtype=np.int32)
    Mydatasets['Patient_Name'] = np.array(Mydatasets['Patient_Name'].values, dtype=np.int32)

    # Merge patch and clinical data on Patient_Name
    Mydatasets_s = pd.merge(Mydatasets, PreRVTS, how="inner", on='Patient_Name')

    return Mydatasets_s, patch_images

# Normalization Function
def normalization(data):
    """
    Apply min-max normalization to data.

    Parameters:
    data (np.ndarray): Input data.

    Returns:
    np.ndarray: Normalized data.
    """
    data_min = data.min()
    data_max = data.max()
    return (data - data_min) / (data_max - data_min)

# Box-Cox Transformation
def BoxCox_Change(data):
    """
    Apply Box-Cox transformation and normalization to data.

    Parameters:
    data (np.ndarray): Input data.

    Returns:
    np.ndarray: Transformed and normalized data.
    """
    data_positive_values = data + 1e-18  # Avoid zero values for Box-Cox
    data_positive_BoxCox, _ = stats.boxcox(data_positive_values)
    return normalization(data_positive_BoxCox)

# Function to Get the Indices of Max and Min Values
def get_max_min_indices(lst, num_points):
    """
    Get indices of maximum and minimum values in a list.

    Parameters:
    lst (list): Input list of values.
    num_points (int): Number of points to retrieve for max and min.

    Returns:
    list: Indices of maximum values.
    list: Indices of minimum values.
    """
    indexed_lst = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    max_indices = [index for index, value in indexed_lst[:int(num_points / 2)]]
    min_indices = [index for index, value in indexed_lst[-int(num_points / 2):]]
    return max_indices, min_indices

# Function to Add Borders Around Images
def Add_Border(image_list, Bold_Image_Size):
    """
    Add a border around a list of images.

    Parameters:
    image_list (list): List of images.
    Bold_Image_Size (int): Size of the border.

    Returns:
    list: List of images with borders.
    """
    New_List_Image = []
    for img in image_list:
        bordered_img = cv2.copyMakeBorder(img, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size, Bold_Image_Size,
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255))
        New_List_Image.append(bordered_img)
    return New_List_Image

# Function to Merge Masked Images into One Large Image
def mask_merge(image_list, image_shape, image_size):
    """
    Merge small images (patches) into one large image.

    Parameters:
    image_list (list): List of small images (patches).
    image_shape (tuple): The dimensions of the merged image.
    image_size (int): Size of each small image.

    Returns:
    np.ndarray: The merged image.
    """
    w_num, h_num = image_shape[0], image_shape[1]
    merged_image = np.zeros((w_num * image_size, h_num * image_size, 3), dtype=float)
    index = 0
    for i in range(w_num - 1):
        for j in range(h_num - 1):
            merged_image[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size] = image_list[index]
            index += 1
        merged_image[i * image_size:(i + 1) * image_size, (j + 1) * image_size:] = \
            image_list[index][:, -(merged_image.shape[1] - (j + 1) * image_size):]
        index += 1
    for k in range(h_num - 1):
        merged_image[-image_size:, k * image_size:(k + 1) * image_size] = image_list[index]
        index += 1
    merged_image[(i + 1) * image_size:, (k + 1) * image_size:] = \
        image_list[index][-(merged_image.shape[0] - (i + 1) * image_size):, -(merged_image.shape[1] - (k + 1) * image_size):]
    return merged_image

# Function to Load Dataset Paths
def Get_Datasets(inds):
    """
    Load paths for datasets and relevant CSV files based on the given index.

    Parameters:
    inds (int): Index indicating which dataset to load (0 or 1).

    Returns:
    DataFrame: Patch data.
    str: Path to HDF5 image file.
    str: Output image path.
    DataFrame: PreRVT data (or None if not applicable).
    """
    if inds == 0:
        # Dataset for index 0
        H5DImagePath = "path/to/Dataset0_Image.hdf5"
        Out_ImagePath = "Output_Image0.png"
        Orial_csvPath = "path/to/Dataset0_Info.csv"
        Weight_csvPath = "path/to/Dataset0_Weight.csv"
        Pre_RVT_Path = "path/to/Dataset0_Result.csv"
    else:
        # Dataset for index 1
        H5DImagePath = "path/to/Dataset1_Image.hdf5"
        Out_ImagePath = "Output_Image1.png"
        Orial_csvPath = "path/to/Dataset1_Info.csv"
        Weight_csvPath = "path/to/Dataset1_Weight.csv"
        Pre_RVT_Path = None

    # Load and merge patch data with attention weight data
    Orial_Csv = pd.read_csv(Orial_csvPath)
    Weight_Csv = pd.read_csv(Weight_csvPath).iloc[:, 1:]
    H5DCSV = pd.merge(Orial_Csv, Weight_Csv, how='inner', on='Patch_Name')[['Unnamed: 0', 'Patch_Name', 'Attention_Weight']]
    
    # Load PreRVT if available
    Pre_RVT = pd.read_csv(Pre_RVT_Path)[['SampleID', 'RVT', 'PreRVT']] if Pre_RVT_Path else None

    return H5DCSV, H5DImagePath, Out_ImagePath, Pre_RVT

# Main Execution Block
if __name__ == '__main__':
    # Load dataset (use index 0 for first dataset)
    H5DCSV, H5DImagePath, Out_ImagePath, Pre_RVT = Get_Datasets(inds=0)
    
    # Apply Box-Cox transformation and normalization to the attention weights
    Weight = H5DCSV['Attention_Weight'].values
    WeightA = BoxCox_Change(Weight)
    H5DCSV['Attention_Weight'] = normalization(WeightA)
    
    # Load patch-level data and images
    Train_Patch, patch_images = Dataset(H5D_File=H5DCSV, H5DImagePath=H5DImagePath, PreRVTS=Pre_RVT)
    Patient_Level = Train_Patch.groupby(by='Patient_Name')

    # Define parameters for image matrix generation
    Row_, COL_ = 10, 5
    nums = Row_ * COL_
    Bold_Linner_Szie = 10
    Cut_Image_Size = 224
    Patient_code = ['575312', '586146', '508956', '713650']  # Specify patients of interest

    # Loop through patients and generate attention heatmaps
    for index, patients_inf in enumerate(Patient_Level):
        patientsname = str(patients_inf[0])
        if patientsname in Patient_code:
            patients = patients_inf[1]
            PreRVT = patients['PreRVT'].values[0]
            types = 'Dig.MPR-' if PreRVT > 0.19349837 else 'Dig.MPR+'

            # Get top and bottom patches based on attention weight
            max_indices, min_indices = get_max_min_indices(patients['Attention_Weight'].values, num_points=nums * 2)
            Patch_In_All_Max = patients['Unnamed: 0'].values[max_indices]
            Patch_In_All_Min = patients['Unnamed: 0'].values[min_indices]

            # Extract max and min attention patches
            max_Images = patch_images[Patch_In_All_Max, :, :, :]
            min_Images = patch_images[Patch_In_All_Min, :, :, :]
            max_Images_list = [max_Images[i] for i in range(Row_ * COL_)]
            min_Images_list = [min_Images[i] for i in range(Row_ * COL_)]

            # Add borders and merge images into a single image
            max_Border_List = Add_Border(max_Images_list, Bold_Image_Szie=Bold_Linner_Szie)
            min_Border_List = Add_Border(min_Images_list, Bold_Image_Szie=Bold_Linner_Szie)
            Border_Size = Cut_Image_Size + Bold_Linner_Szie * 2
            max_Imagess = mask_merge(max_Border_List, [COL_, Row_], Border_Size)
            min_Imagess = mask_merge(min_Border_List, [COL_, Row_], Border_Size)

            # Visualize and save max attention image
            plt.figure()
            plt.imshow(max_Imagess / 255.0)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"output_directory/{types}_High_{patientsname}_{Out_ImagePath}", dpi=1500, bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()

            # Visualize and save min attention image
            plt.figure()
            plt.imshow(min_Imagess / 255.0)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"output_directory/{types}_Low_{patientsname}_{Out_ImagePath}", dpi=1500, bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()
