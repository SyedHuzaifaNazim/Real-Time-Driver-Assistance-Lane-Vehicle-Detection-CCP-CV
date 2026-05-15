import cv2
import numpy as np
from scipy.signal import convolve2d
import os

INPUT_VIDEO = "input_video.mp4"
OUTPUT_DIR = "gradient_maps"

def manual_sobel(image):
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define Sobel kernels
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
                   
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    
    # Convolve
    Gx = convolve2d(image, Kx, mode='same', boundary='symm')
    Gy = convolve2d(image, Ky, mode='same', boundary='symm')
    
    # Calculate magnitude
    magnitude = np.sqrt(Gx**2 + Gy**2)
    
    # Normalize to 0-255
    magnitude = np.clip(magnitude, 0, 255)
    magnitude = magnitude.astype(np.uint8)
    
    return magnitude

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error opening video file {INPUT_VIDEO}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 10:
        print("Video is too short.")
        return
        
    # Sample 10 equidistant frames
    sample_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    
    for i, idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        print(f"Processing frame {idx} for gradient map...")
        grad_map = manual_sobel(frame)
        
        output_path = os.path.join(OUTPUT_DIR, f"frame_{idx}_gradient.jpg")
        cv2.imwrite(output_path, grad_map)
        
    cap.release()
    print(f"Saved 10 gradient maps to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()