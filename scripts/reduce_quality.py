import os
from PIL import Image

def reduce_image_quality(input_dir, output_dir, quality=50):
    """
    Reduces the quality of all JPEG images in the specified directory and saves them to the output directory.
    
    Args:
    input_dir (str): Directory containing the images to process.
    output_dir (str): Directory where the processed images will be saved.
    quality (int): The quality level of the saved images, between 1 (worst) and 95 (best).
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    count = 0
    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            count += 1
            file_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Open the image
            with Image.open(file_path) as img:
                # Convert image to RGB to avoid issues if it's in a different mode
                img = img.convert('RGB')
                # Save the image with reduced quality
                img.save(output_path, 'JPEG', quality=quality)
            # print n out of total_n
            print(f"{count} completed")

# Example usage
reduce_image_quality('/Users/edbayes/Desktop/brazil/molra', '/Users/edbayes/Desktop/brazil/molra', quality=50)