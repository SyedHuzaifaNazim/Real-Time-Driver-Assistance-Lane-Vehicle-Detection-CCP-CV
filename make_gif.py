import cv2
from PIL import Image

def create_gif(input_video, output_gif, start_sec=0, duration_sec=5, fps=10):
    print(f"Opening {input_video}...")
    cap = cv2.VideoCapture(input_video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30
        
    start_frame = int(start_sec * video_fps)
    end_frame = int((start_sec + duration_sec) * video_fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    frame_count = start_frame
    
    frame_skip = int(video_fps / fps)
    if frame_skip < 1:
        frame_skip = 1
        
    print(f"Extracting frames from {start_sec}s to {start_sec + duration_sec}s...")
    while cap.isOpened() and frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            # Resize to keep gif size reasonable (640x360)
            frame = cv2.resize(frame, (640, 360))
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            frames.append(img)
            
        frame_count += 1
        
    cap.release()
    
    if frames:
        print(f"Saving {len(frames)} frames as {output_gif} (this might take a few seconds)...")
        frames[0].save(output_gif, format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=int(1000/fps), loop=0)
        print(f"Successfully saved {output_gif}")
    else:
        print("Failed to read frames. Video might be too short or missing.")

if __name__ == '__main__':
    # Start at 5 seconds in to get some good action, capture 5 seconds of footage at 10 frames per second
    create_gif('output_video.mp4', 'output_demo.gif', start_sec=5, duration_sec=5, fps=10)
