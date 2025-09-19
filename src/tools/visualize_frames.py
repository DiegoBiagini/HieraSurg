import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

def create_image_grid(folder_path, output_path):
    # Get list of image files in the folder and sort them
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))])
    
    # Check if there are exactly 16 images
    if len(image_files) != 16:
        print("The folder must contain exactly 16 images.")
        return
    
    # Load images
    images = [Image.open(os.path.join(folder_path, img_file)) for img_file in image_files]
    
    # Create a figure to display the images
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    # Plot each image in the grid
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')  # Hide the axes
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path)
    print(f"Grid image saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <folder_path> <output_path>")
    else:
        folder_path = sys.argv[1]
        output_path = sys.argv[2]
        create_image_grid(folder_path, output_path)