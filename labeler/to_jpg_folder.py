import os
from PIL import Image
import argparse

def mirror_folder_structure(src_folder, dst_folder):
    """
    Mirror the folder structure and convert PNG images to JPG in the destination folder.
    
    :param src_folder: Path to the source folder containing PNG images.
    :param dst_folder: Path to the destination folder where JPG images will be saved.
    """
    # Walk through the source folder
    for root, dirs, files in os.walk(src_folder):
        print("test")
        # Create corresponding directories in the destination folder
        rel_path = os.path.relpath(root, src_folder)  # Relative path to maintain folder structure
        dst_dir = os.path.join(dst_folder, rel_path)
        os.makedirs(dst_dir, exist_ok=True)  # Create the directory if it doesn't exist
        
        for file in files:
            if file.lower().endswith('.png'):  # Process only PNG files
                # Construct full file paths
                src_file_path = os.path.join(root, file)
                # Change the extension to .jpg
                new_file_name = os.path.splitext(file)[0] + '.jpg'
                dst_file_path = os.path.join(dst_dir, new_file_name)
                
                # Open the image and convert to JPG
                with Image.open(src_file_path) as img:
                    rgb_img = img.convert('RGB')  # Convert to RGB (since JPG doesn't support alpha channels)
                    rgb_img.save(dst_file_path, 'JPEG')
                
                print(f"Converted: {src_file_path} -> {dst_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirror folder structure and convert PNG images to JPG.")
    parser.add_argument('source_folder', type=str, help="Path to the source folder containing PNG images")
    parser.add_argument('destination_folder', type=str, help="Path to the destination folder where JPG images will be saved")
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    mirror_folder_structure(args.source_folder, args.destination_folder)