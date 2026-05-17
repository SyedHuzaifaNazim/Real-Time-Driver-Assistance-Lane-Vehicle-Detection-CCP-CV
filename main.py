import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- Configuration ---
INPUT_VIDEO = "input_video.mp4"
OUTPUT_VIDEO = "output_video.mp4"
MODEL_PATH = "yolov8n.pt"

# Classes to track: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]
PROXIMITY_THRESHOLD_RATIO = 0.15  # Area ratio threshold for "Too Close" warning

# Dictionary to store vehicle trajectory histories
track_history = defaultdict(lambda: [])

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lanes(img, lines, color=[0, 255, 0], thickness=6):
    # Separate lines into left and right lanes
    left_lines = []
    right_lines = []
    left_weights = []
    right_weights = []
    
    if lines is None:
        return img, "Lane Missing"

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            
            if slope < -0.3:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            elif slope > 0.3:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    lane_img = np.zeros_like(img)
    lane_status = "Lane Clear"

    y1 = img.shape[0]
    y2 = int(y1 * 0.6) # Horizon line
    
    # Left Lane
    if len(left_lines) > 0:
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights)
        slope, intercept = left_lane
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        cv2.line(lane_img, (x1, y1), (x2, y2), color, thickness)
        if x1 > img.shape[1] // 2: # Left line crossed middle
            lane_status = "Lane Departure Warning!"
    
    # Right Lane
    if len(right_lines) > 0:
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights)
        slope, intercept = right_lane
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        cv2.line(lane_img, (x1, y1), (x2, y2), color, thickness)
        if x1 < img.shape[1] // 2: # Right line crossed middle
            lane_status = "Lane Departure Warning!"

    img = cv2.addWeighted(img, 1.0, lane_img, 1.0, 0.0)
    return img, lane_status

def process_lane_detection(frame):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Define Region of Interest (Trapezoid)
    region_of_interest_vertices = [
        (int(width * 0.1), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.9), height)
    ]
    
    cropped_edges = region_of_interest(edges, np.array([region_of_interest_vertices], np.int32))
    
    # Hough Transform
    lines = cv2.HoughLinesP(cropped_edges, rho=2, theta=np.pi/180, threshold=50,
                            lines=np.array([]), minLineLength=40, maxLineGap=100)
    
    return draw_lanes(frame, lines)

def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error opening video file {INPUT_VIDEO}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_area = width * height
    vehicle_count = 0
    counting_line_y = int(height * 0.7) # Line to cross for counting
    counted_ids = set()
    
    speed_estimate = 10 # Base simulated speed
    frame_count = 0

    print("Starting processing...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # 1. Lane Detection
        frame, lane_status = process_lane_detection(frame)
        
        # 2. Vehicle Detection and Tracking
        results = model.track(frame, persist=True, classes=VEHICLE_CLASSES, verbose=False)
        
        too_close_warning = False
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, cls in zip(boxes, track_ids, clss):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                # Check proximity
                if area > frame_area * PROXIMITY_THRESHOLD_RATIO:
                    too_close_warning = True
                    box_color = (0, 0, 255) # Red for danger
                else:
                    box_color = (255, 0, 0) # Blue for normal
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Trailing logic
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                track_history[track_id].append((center_x, center_y))
                if len(track_history[track_id]) > 30: # Retain last 30 frames
                    track_history[track_id].pop(0)
                
                points = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)
                
                # Counting logic
                if len(track_history[track_id]) >= 2:
                    prev_y = track_history[track_id][-2][1]
                    # If moving downwards across the counting line
                    if prev_y < counting_line_y and center_y >= counting_line_y:
                        if track_id not in counted_ids:
                            vehicle_count += 1
                            counted_ids.add(track_id)
        
        # 3. HUD Overlay
        # Simulate slight speed variation
        if frame_count % int(fps) == 0:
            speed_estimate += np.random.randint(-2, 3)
            speed_estimate = max(0, speed_estimate)
        
        # Draw Counting Line
        cv2.line(frame, (0, counting_line_y), (width, counting_line_y), (255, 255, 0), 2)
        
        # HUD Text
        cv2.putText(frame, f"Speed: {speed_estimate} km/h", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles Count: {vehicle_count}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        lane_color = (0, 0, 255) if "Warning" in lane_status else (0, 255, 0)
        cv2.putText(frame, f"Status: {lane_status}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, lane_color, 2)
        
        if too_close_warning:
            cv2.putText(frame, "WARNING: VEHICLE TOO CLOSE!", (width // 2 - 250, height - 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        out.write(frame)

    cap.release()
    out.release()
    print("Processing complete. Output saved to", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
