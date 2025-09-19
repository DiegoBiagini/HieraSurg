import os
import argparse

def remove_unmatched_frames(video_folder, mask_folder):
    global tot
    # Get list of frame files and mask files
    frame_files = set(os.listdir(video_folder))
    mask_files = set(os.listdir(mask_folder))

    # Extract the base names without extensions
    frame_basenames = {f"{int(os.path.splitext(f)[0])//8:06d}" for f in frame_files}
    mask_basenames = {os.path.splitext(f)[0] for f in mask_files}

    # Find frames that do not have a corresponding mask
    unmatched_frames = frame_basenames - mask_basenames

    # Remove unmatched frames
    for frame in unmatched_frames:
        for i in range(8):
            frame_path = os.path.join(video_folder, f"{int(frame)*8+i:06d}" + '.jpg')
            if os.path.exists(frame_path):
                os.remove(frame_path)
                #print(f'Removed {frame_path}')

def process_folders(base_folder):
    for folder_name in os.listdir(base_folder):
        if folder_name.startswith('video') and not folder_name.endswith('_masks'):
            video_folder = os.path.join(base_folder, folder_name)
            mask_folder = os.path.join(base_folder, folder_name + '_masks')
            if os.path.isdir(video_folder) and os.path.isdir(mask_folder):
                remove_unmatched_frames(video_folder, mask_folder)

def main():
    parser = argparse.ArgumentParser(description="Remove frames that do not have a corresponding mask file.")
    parser.add_argument('base_folder', type=str, help='Path to the base folder containing video and mask folders.')
    
    args = parser.parse_args()
    process_folders(args.base_folder)

if __name__ == "__main__":
    main()