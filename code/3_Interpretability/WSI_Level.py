import tables
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy import stats
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Helper function to truncate patient names if necessary
def nameins(data):
    """
    Shorten the patient name if it contains '-' or is longer than 7 characters.
    """
    res = data.split("-")[0] if len(data.split("-")) > 1 else data
    return res[:7] if len(res) > 7 else res

# Dataset loader function
def Dataset(H5D_File, H5DImagePath, PreRVTS):
    """
    Load patch images and clinical data, then merge them based on patient names.
    
    Parameters:
    H5D_File (DataFrame): Patch-level data
    H5DImagePath (str): Path to the HDF5 file containing patch images
    PreRVTS (DataFrame): Clinical data with RVT and PreRVT values
    
    Returns:
    DataFrame, HDF5 dataset: Merged dataset and corresponding patch images
    """
    store = tables.open_file(H5DImagePath, mode='r')
    
    # Processing the patch-level data
    H5D_File['Name'] = [name.split("_")[0] + '_' + name.split("_")[-1] for name in H5D_File['Patch_Name'].values]
    Mydatasets = H5D_File
    Mydatasets['Patient_Name'] = [name.split("_")[0] for name in Mydatasets['Name'].values]
    Mydatasets['Coordinates'] = [name.split("_")[1] for name in Mydatasets['Name'].values]
    Mydatasets['Index_Image'] = np.arange(0, Mydatasets.shape[0])
    
    patch_images = store.root.patches

    # Merging with clinical data
    PreRVTS.columns = ['Patient_Name', 'RVT', 'PreRVT']
    PreRVTS['Patient_Name'] = np.array(PreRVTS['Patient_Name'].values, dtype=str)
    Mydatasets['Patient_Name'] = np.array(Mydatasets['Patient_Name'].values, dtype=str)
    
    # Merge datasets
    Mydatasets_s = pd.merge(Mydatasets, PreRVTS, how="inner", on='Patient_Name')
    
    # If merging fails, apply name shortening and try again
    if len(Mydatasets_s) == 0:
        Mydatasets['Patient_Name'] = Mydatasets['Patient_Name'].apply(nameins)
        Mydatasets_s = pd.merge(Mydatasets, PreRVTS, how="inner", on='Patient_Name')
    
    return Mydatasets_s, patch_images

# Function to generate a Whole Slide Image (WSI) from patches
def generate_wsi_image(patch_dir, patients, Image_Resize, border_size):
    """
    Create a WSI image by assembling patches into a larger image based on coordinates.
    
    Parameters:
    patch_dir (HDF5): HDF5 dataset containing the image patches
    patients (DataFrame): DataFrame containing the patch information for a patient
    Image_Resize (int): Size to resize patches
    border_size (int): Size of the border between patches
    
    Returns:
    np.ndarray: The resulting WSI image
    """
    patches = []
    max_x, max_y = 0, 0
    
    # Loop through patches and collect images
    for index, filename in enumerate(patients['Coordinates']):
        patch_coords = filename.split('-')
        x, y = int(patch_coords[0]), int(patch_coords[1])
        Image_index_ = patients['Unnamed: 0'].values[index]
        patch = patch_dir[Image_index_]
        rgb_image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        bgr_images = cv2.resize(bgr_image, (Image_Resize, Image_Resize))
        patches.append((x, y, bgr_images))
        max_x, max_y = max(max_x, x), max(max_y, y)
    
    # Create a blank WSI image
    patch_size = Image_Resize
    wsi_width = (max_x + 1) * (patch_size + border_size) + border_size
    wsi_height = (max_y + 1) * (patch_size + border_size) + border_size
    wsi_image = np.zeros((wsi_height, wsi_width, 3), dtype=np.uint8) + 242  # Light gray background
    
    # Place patches into the WSI image
    for x, y, patch in patches:
        start_x = x * (patch_size + border_size) + border_size
        start_y = y * (patch_size + border_size) + border_size
        wsi_image[start_y:start_y + patch_size, start_x:start_x + patch_size, :] = patch
    
    return wsi_image

# Min-max normalization
def normalization(data):
    """
    Apply min-max normalization to the input data.
    
    Parameters:
    data (np.ndarray): Input data
    
    Returns:
    np.ndarray: Normalized data
    """
    data_min = data.min()
    data_max = data.max()
    return (data - data_min) / (data_max - data_min)

# Box-Cox transformation
def BoxCox_Change(data):
    """
    Apply Box-Cox transformation to the input data and normalize it.
    
    Parameters:
    data (np.ndarray): Input data
    
    Returns:
    np.ndarray: Transformed and normalized data
    """
    data_positive_values = data + 1e-18  # Ensure no zero values
    data_positive_BoxCox, _ = stats.boxcox(data_positive_values)
    return normalization(data_positive_BoxCox)

# Function to generate an attention-based weight image
def generate_weight_image(patients, Image_Resize, border_size):
    """
    Create an image visualizing attention weights for each patch.
    
    Parameters:
    patients (DataFrame): DataFrame containing patch and attention data
    Image_Resize (int): Size to resize patches
    border_size (int): Size of the border between patches
    
    Returns:
    np.ndarray: The resulting attention weight image
    """
    patches = []
    max_x, max_y = 0, 0
    Weight_List = patients['Attention_Weight'].values
    
    # Loop through patches and create color-mapped images based on attention weights
    for index, filename in enumerate(patients['Coordinates']):
        patch_coords = filename.split('-')
        x = int(patch_coords[0])
        y = int(patch_coords[1])
        weight = Weight_List[index]
        
        # Generate color mapping for the weight
        colormap = cm.get_cmap('magma')
        color_matrix = colormap(weight)
        image = np.tile(color_matrix[:3], (224, 224, 1))
        image = (image * 255).astype(np.uint8)
        
        bgr_images = cv2.resize(image, (Image_Resize, Image_Resize))
        patches.append((x, y, bgr_images))
        max_x, max_y = max(max_x, x), max(max_y, y)
    
    # Create a blank image for the weights visualization
    patch_size = Image_Resize
    wsi_width = (max_x + 1) * (patch_size + border_size) + border_size
    wsi_height = (max_y + 1) * (patch_size + border_size) + border_size
    wsi_image = np.zeros((wsi_height, wsi_width, 3), dtype=np.uint8) + 255  # White background
    
    # Place patches into the WSI weight image
    for x, y, patch in patches:
        start_x = x * (patch_size + border_size) + border_size
        start_y = y * (patch_size + border_size) + border_size
        wsi_image[start_y:start_y + patch_size, start_x:start_x + patch_size, :] = patch
    
    return wsi_image

# Function to load dataset paths and data based on an index
def Get_Datasets(inds):
    """
    Load dataset paths and CSV files based on the index (0 or 1).
    
    Parameters:
    inds (int): Index to select between datasets (0 or 1)
    
    Returns:
    tuple: Dataset files and paths
    """
    if inds == 0:
        # Dataset for index 0 (External_XH)
        H5DImagePath = "path/to/Dataset0_Image.hdf5"
        Out_ImagePath = "Output_Image0.png"
        Orial_csvPath = "path/to/Dataset0_Info.csv"
        Weight_csvPath = "path/to/Dataset0_Weight.csv"
        Pre_RVT_Path = "path/to/Dataset0_Result.csv"
    else:
        # Dataset for index 1 (Discovery_BZ)
        H5DImagePath = "path/to/Dataset1_Image.hdf5"
        Out_ImagePath = "Output_Image1.png"
        Orial_csvPath = "path/to/Dataset1_Info.csv"
        Weight_csvPath = "path/to/Dataset1_Weight.csv"
        Pre_RVT_Path = "path/to/Dataset1_Result.csv"

    # Load the CSV files and merge patch data with weight data
    Orial_Csv = pd.read_csv(Orial_csvPath)
    Orial_Csv['Patch_Name'] = [name.split("_")[0] + "_0_" + name.split("_")[-1] for name in Orial_Csv['Patch_Name'].values]
    Weight_Csv = pd.read_csv(Weight_csvPath).iloc[:, 1:]
    Weight_Csv['Patch_Name'] = [name.split("_")[0] + "_0_" + name.split("_")[-1] for name in Weight_Csv['Patch_Name'].values]
    H5DCSV = pd.merge(Orial_Csv, Weight_Csv, how='inner', on='Patch_Name')[['Unnamed: 0', 'Patch_Name', 'Attention_Weight']]
    Pre_RVT = pd.read_csv(Pre_RVT_Path)[['SampleID', 'RVT', 'PreRVT']]

    return H5DCSV, H5DImagePath, Out_ImagePath, Pre_RVT

# Main execution block
if __name__ == '__main__':
    inds = 0  # 0 for dataset 0, 1 for dataset 1
    H5DCSV, H5DImagePath, Out_ImagePath, Pre_RVT = Get_Datasets(inds)
    
    # Apply Box-Cox transformation and normalization to the attention weights
    Weight = H5DCSV['Attention_Weight'].values
    WeightA = BoxCox_Change(Weight)
    H5DCSV['Attention_Weight'] = normalization(WeightA)
    
    # Load patch-level data and images
    Train_Patch, patch_images = Dataset(H5D_File=H5DCSV, H5DImagePath=H5DImagePath, PreRVTS=Pre_RVT)
    Patient_Level = Train_Patch.groupby(by='Patient_Name')
    Patient_code = ['']  # Specify patient code(s) here
    
    # Loop through patients and generate weight images for specific patient codes
    for index, patients_inf in enumerate(Patient_Level):
        patientsname = str(patients_inf[0])
        if patientsname in Patient_code:
            print(f"\033[31m------------------------- Generating attention heatmap for patient {patientsname} -------------------------\033[0m")
            patients = patients_inf[1]
            PreRVT = patients['PreRVT'].values[0]
            Cutoffs = 0.19349837
            types = str(PreRVT)
            
            # Generate and save weight image
            Weight_Image = generate_weight_image(patients, Image_Resize=256, border_size=0)
            plt.figure()
            plt.imshow(Weight_Image)
            plt.axis('off')  # Hide axis
            plt.tight_layout()
            plt.savefig(f"output_directory/FOB_{types}_{patientsname}_Weight{Out_ImagePath}", dpi=1500, bbox_inches='tight', pad_inches=0)
            plt.show()
            print(f"\033[34m------------- Successfully generated attention heatmap for patient {patientsname} -------------\033[0m")
            plt.close()
