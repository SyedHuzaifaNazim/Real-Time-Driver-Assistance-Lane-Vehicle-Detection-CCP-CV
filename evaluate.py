import cv2
import numpy as np
from ultralytics import YOLO

INPUT_VIDEO = "input_video.mp4"
MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]

def process_lane_detection_eval(frame):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    region_of_interest_vertices = [
        (int(width * 0.1), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.9), height)
    ]
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([region_of_interest_vertices], np.int32), 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(cropped_edges, rho=2, theta=np.pi/180, threshold=50,
                            lines=np.array([]), minLineLength=40, maxLineGap=100)
    
    return lines is not None and len(lines) > 0

def main():
    print("Loading YOLO model for evaluation...")
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error opening video file {INPUT_VIDEO}")
        return
        
    frames_to_eval = 50
    lane_success_count = 0
    total_vehicles_detected = 0
    
    print(f"Evaluating {frames_to_eval} frames...")
    
    for i in range(frames_to_eval):
        ret, frame = cap.read()
        if not ret:
            print(f"Video ended early at frame {i}")
            frames_to_eval = i
            break
            
        # 1. Evaluate Lane Detection (Success = found at least 1 line segment in ROI)
        lane_found = process_lane_detection_eval(frame)
        if lane_found:
            lane_success_count += 1
            
        # 2. Evaluate Vehicle Detection
        results = model.predict(frame, classes=VEHICLE_CLASSES, verbose=False)
        if results[0].boxes is not None:
            total_vehicles_detected += len(results[0].boxes)
            
    cap.release()
    
    if frames_to_eval > 0:
        lane_success_rate = (lane_success_count / frames_to_eval) * 100
        avg_vehicles_per_frame = total_vehicles_detected / frames_to_eval
        
        print("\n=== Evaluation Results ===")
        print(f"Frames Evaluated: {frames_to_eval}")
        print(f"Lane Detection Success Rate: {lane_success_rate:.2f}% (Found lanes in {lane_success_count}/{frames_to_eval} frames)")
        print(f"Average Vehicles Detected per Frame: {avg_vehicles_per_frame:.2f}")
    else:
        print("No frames evaluated.")

if __name__ == "__main__":
    main()
