import cv2
import time
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(
    engine_path: str,
    video_path: str,
    output_path: str = "output.mp4",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    show_fps: bool = True,
    display: bool = True
):
    """Process video with TensorRT model using Ultralytics YOLO"""
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize YOLO model
    model = YOLO(engine_path)
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps_list = []
    
    # Pause state
    paused = False

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Total frames: {total_frames}")
    logger.info("Controls:")
    logger.info("  'p' - Pause/Resume")
    logger.info("  'q' - Quit")
    cv2.namedWindow('YOLO Inference', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLO Inference', 1920, 1080)
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference
            #gray
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            results = model(frame, verbose=False, imgsz=416)
            
            # Get detections
            detections = results[0].boxes.data.cpu().numpy()
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf > conf_threshold:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Class {int(cls)} ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update FPS every 30 frames
                current_time = time.time()
                elapsed_time = current_time - start_time
                current_fps = 30 / elapsed_time
                fps_list.append(current_fps)
                start_time = current_time

            if show_fps and len(fps_list) > 0:
                avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])  # Average of last 30 FPS measurements
                cv2.putText(
                    frame,
                    f"FPS: {avg_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            # Write frame to output video
            out.write(frame)

        # Display frame if requested
        if display:
            # Add pause indicator
            if paused:
                cv2.putText(
                    frame,
                    "PAUSED",
                    (width // 2 - 50, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            
            cv2.imshow('YOLO Inference', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                logger.info("Paused" if paused else "Resumed")

        # Log progress
        if not paused and frame_count % 100 == 0:
            logger.info(f"Processed {frame_count}/{total_frames} frames")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculate and log final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    logger.info("Processing complete!")
    logger.info(f"Average FPS: {avg_fps:.2f}")
    logger.info(f"Output saved to: {output_path}")

def main():
    # Example usage
    engine_path = r"runs\detect\train6\weights\best.onxx"
    video_path = r"C:\Users\darke\Documents\python\Yolosu\videos\Songs Compilation VI_Collab Extra.mp4"
    
    process_video(
        engine_path=engine_path,
        video_path=video_path,
        output_path="output.mp4",
        conf_threshold=0.25,
        iou_threshold=0.45,
        show_fps=True,
        display=True
    )

if __name__ == "__main__":
    main() 