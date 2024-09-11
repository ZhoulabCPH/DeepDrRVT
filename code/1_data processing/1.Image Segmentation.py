# Import necessary libraries
from histolab.slide import Slide  # For handling whole slide images (WSI)
from histolab.tiler import GridTiler  # For extracting tiles from the WSI
import shutil  # To handle file operations like removing directories
import os  # For handling file and directory operations


# Define the function for tiling whole slide images (WSI)
def tiling_WSI():
    # Initialize the GridTiler object with specified parameters
    grid_tiles_extractor = GridTiler(
        tile_size=(224, 224),  # Size of each tile in pixels
        level=0,  # Pyramid level to extract from (0 is the highest resolution)
        check_tissue=True,  # Check if the tile contains tissue
        tissue_percent=40,  # Minimum percentage of tissue required in the tile
        pixel_overlap=0,  # Overlap between tiles
        prefix="",  # Prefix for the output tile filenames
        suffix=".png"  # Suffix/extension for the output tile filenames
    )

    count = 0  # Counter for processed slides
    path = ""  # Define the path where the WSIs are stored (to be filled in)

    # Loop through each slide in the specified directory
    for slide in os.listdir(path):
        print(f'Tiling {slide}.')  # Print the name of the current slide being processed

        WSI_path = path + slide  # Complete file path of the current slide
        output_path = ""  # Define the path to store output tiles (to be filled in)

        # Check if the output directory exists
        if os.path.exists(output_path):
            # If it exists, skip the current slide
            continue

            # If the directory exists, remove it to prevent overwriting issues
            shutil.rmtree(output_path)
            print("Directory already exists. It has been deleted.")

        else:
            # If the output directory doesn't exist, create it
            os.mkdir(output_path)
            print(f"Created directory: {output_path}")

        # Load the slide using the histolab Slide class
        slide1 = Slide(WSI_path, output_path)

        # Extract tiles from the slide using the GridTiler
        grid_tiles_extractor.extract(slide1)

        # Increment the count and indicate that the tiling for this slide is complete
        count += 1
        print(f"{str(count)} slides processed!")


# Run the tiling function
tiling_WSI()
