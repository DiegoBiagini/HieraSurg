import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

def is_mostly_black(frame, threshold=0.95):
    return False
    return np.mean(frame < 10) > threshold

def extract_frames(input_folder, output_folder, framerate):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in os.listdir(input_folder):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(input_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_video_folder = os.path.join(output_folder, video_name)
            
            if not os.path.exists(output_video_folder):
                os.makedirs(output_video_folder)

            cap = cv2.VideoCapture(video_path)
            #cap.set(cv2.CAP_PROP_FPS,24)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            print(video_fps)

            frame_interval = int(video_fps / framerate)

            frame_count = 0
            saved_count = 0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            with tqdm(total=total_frames, desc=f"Processing {video_file}") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        if not is_mostly_black(frame):
                            output_path = os.path.join(output_video_folder, f"{saved_count:06d}.jpg")
                            cv2.imwrite(output_path, frame)
                            saved_count += 1

                    frame_count += 1
                    pbar.update(1)

            cap.release()
            print(f"Extracted {saved_count} frames from {video_file}")

def extract_frames_fix(input_folder, output_folder, framerate):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in os.listdir(input_folder):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(input_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_video_folder = os.path.join(output_folder, video_name)
            
            if not os.path.exists(output_video_folder):
                os.makedirs(output_video_folder)

            cap = cv2.VideoCapture(video_path)
            #cap.set(cv2.CAP_PROP_FPS,24)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            print(video_fps)

            frame_interval = int(video_fps / framerate)

            frame_count = 0
            saved_count = 0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            with tqdm(total=total_frames, desc=f"Processing {video_file}") as pbar:
                sec_counter = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if sec_counter == (video_fps-1):
                        sec_counter = 0
                    else:
                        if (frame_count%video_fps) % frame_interval == 0:
                            #print("sec counter ", sec_counter) 
                            if not is_mostly_black(frame):
                                output_path = os.path.join(output_video_folder, f"{saved_count:06d}.jpg")
                                cv2.imwrite(output_path, frame)
                                #print(frame_count)
                                saved_count += 1                        
                        sec_counter += 1
                    frame_count += 1
                    
                    pbar.update(1)

            cap.release()
            print(f"Extracted {saved_count} frames from {video_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos at specified framerate.")
    parser.add_argument("input_folder", help="Path to the folder containing input videos")
    parser.add_argument("output_folder", default="extracted_frames", help="Path to the folder where extracted frames will be saved")
    parser.add_argument("framerate", type=float, default=1.0, help="Framerate at which to extract frames")
    parser.add_argument("--with_fix", action='store_true', help="Include this flag to apply the 25 fps fix")


    args = parser.parse_args()
    if args.with_fix:
        extract_frames_fix(args.input_folder, args.output_folder, args.framerate)
    else:
        extract_frames(args.input_folder, args.output_folder, args.framerate)
    

if __name__ == "__main__":
    main()
