import cv2
from detector import YOLODetector
from tracker import MovementTracker
from video_processor import VideoProcessor

video_path = 'shenzhencar.mp4'

processor = VideoProcessor(video_path)
detector = YOLODetector(model_name='yolov8n.pt')
tracker = MovementTracker(line_ratio=0.9)  # 检测线在距离底部10%处

frame_idx = 0

print("Vehicle counting started...")
print(f"Detection line position: {tracker.line_position * 100}% from top (10% from bottom)")
print("Counting rules: From middle to bottom = EXIT, From bottom to middle = ENTER")

while True:
    ret, frame = processor.read_frame()
    if not ret:
        break

    detections = detector.detect(frame)
    total_count = tracker.update(frame, detections)  # 接收总计数

    # 在视频上画计数
    cv2.putText(frame, f"ENTER: {tracker.enter_count}", (20, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(frame, f"EXIT: {tracker.exit_count}", (20, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    # 显示总车辆数
    cv2.putText(frame, f"TOTAL: {total_count}", (20, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    
    # 添加检测线说明
    if hasattr(tracker, 'line_y'):
        cv2.putText(frame, f"DETECTION LINE: 10% from bottom (Y={tracker.line_y})", 
                   (20, frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    processor.write_frame(frame)

    if frame_idx % 20 == 0:
        print(f"[Frame {frame_idx}] ENTER: {tracker.enter_count}, EXIT: {tracker.exit_count}, TOTAL: {total_count}")
    frame_idx += 1

processor.release()
print(f"✅ Done! ENTER: {tracker.enter_count}, EXIT: {tracker.exit_count}, TOTAL Vehicles: {total_count}") 