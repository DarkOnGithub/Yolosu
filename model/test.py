# from ultralytics import YOLO
# import cv2
# import numpy as np
# from filterpy.kalman import KalmanFilter
# from scipy.optimize import linear_sum_assignment
# import time
# import dxcam
# import win32api
# import win32con
# import math
# from collections import defaultdict, deque
# import threading
# from queue import Queue, Empty
# import statistics
# import torch
# from numba import jit, float32, int32, boolean
# # --- Configuration ---
# # Performance
# YOLO_MODEL_PATH = r"runs\detect\train4\weights\best.onnx" # ONNX model path
# USE_GPU = True  # Set to True to use GPU, False for CPU
# FP16_INFERENCE = True # Use FP16 for faster inference on compatible GPUs
# INFERENCE_SIZE = 416  # Smaller can be faster but less accurate. 320, 416, 640 are common.
# CONFIDENCE_THRESHOLD = 0.30 # Lower for more detections, higher for fewer but more confident
# DXCAM_FPS_TARGET = 60 # Target FPS for screen capture

# # Tracking
# MAX_AGE = 10          # Max frames a track can exist without an update
# MIN_HITS_TO_CONFIRM = 3 # Min hits to confirm a track (reduces false positives)
# IOU_THRESHOLD = 0.2   # Lower for stricter matching

# # Clicking Logic
# CLICK_COOLDOWN = 0.05  # Minimum seconds between clicks
# # Advanced Click Detection Parameters
# PREDICTION_FRAMES = 3  # Number of frames to predict ahead
# MIN_CONFIDENCE_FOR_CLICK = 0.4  # Minimum confidence to consider clicking
# SPEED_WEIGHT = 0.6  # Weight for speed in click decision
# DISTANCE_WEIGHT = 0.4  # Weight for distance in click decision
# MAX_PREDICTION_ERROR = 20  # Maximum allowed prediction error in pixels
# MIN_SPEED_FOR_PREDICTION = 2  # Minimum speed to use prediction
# BASE_CLICK_RADIUS = 20  # Base click radius in pixels
# SPEED_MULTIPLIER = 0.8  # How much speed increases click radius
# SIZE_MULTIPLIER = 0.5  # How much object size affects click radius

# # Debugging
# SHOW_FPS_STATS_INTERVAL = 1.0 # Seconds

# # --- KalmanBoxTracker (largely unchanged, it's standard) ---
# class KalmanBoxTracker:
#     count = 0
#     def __init__(self, bbox):
#         self.kf = KalmanFilter(dim_x=7, dim_z=4)
#         self.kf.F = np.array([[1,0,0,0,1,0,0],
#                              [0,1,0,0,0,1,0],
#                              [0,0,1,0,0,0,1],
#                              [0,0,0,1,0,0,0],
#                              [0,0,0,0,1,0,0],
#                              [0,0,0,0,0,1,0],
#                              [0,0,0,0,0,0,1]])
#         self.kf.H = np.array([[1,0,0,0,0,0,0],
#                              [0,1,0,0,0,0,0],
#                              [0,0,1,0,0,0,0],
#                              [0,0,0,1,0,0,0]])
#         self.kf.R[2:,2:] *= 10.
#         self.kf.P[4:,4:] *= 1000.
#         self.kf.P *= 10.
#         self.kf.Q[-1,-1] *= 0.01
#         self.kf.Q[4:,4:] *= 0.01
#         self.kf.x[:4] = convert_bbox_to_z(bbox).reshape(4, 1)
#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = []
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0

#     def update(self, bbox):
#         self.time_since_update = 0
#         self.history = []
#         self.hits += 1
#         self.hit_streak += 1
#         self.kf.update(convert_bbox_to_z(bbox).reshape(4, 1))

#     def predict(self):
#         if(self.kf.x[6]+self.kf.x[2])<=0:
#             self.kf.x[6] *= 0.0
#         self.kf.predict()
#         self.age += 1
#         if(self.time_since_update>0):
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(convert_x_to_bbox(self.kf.x.flatten()))
#         return self.history[-1]

#     def get_state(self):
#         return convert_x_to_bbox(self.kf.x.flatten())

# @jit(nopython=True)
# def iou(bb_test, bb_gt):
#     """Calculate IOU between two bounding boxes."""
#     xx1 = max(bb_test[0], bb_gt[0])
#     yy1 = max(bb_test[1], bb_gt[1])
#     xx2 = min(bb_test[2], bb_gt[2])
#     yy2 = min(bb_test[3], bb_gt[3])
#     w = max(0., xx2 - xx1)
#     h = max(0., yy2 - yy1)
#     wh = w * h
#     o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
#               + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-7)
#     return o

# @jit(nopython=True)
# def calculate_movement(prev_center, curr_center):
#     """Calculate movement between two points."""
#     return math.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)

# @jit(nopython=True)
# def convert_bbox_to_z(bbox):
#     """Convert bounding box to state vector."""
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w/2.
#     y = bbox[1] + h/2.
#     s = w * h    #scale is area
#     r = w / float(h) #aspect ratio
#     return np.array([x, y, s, r])

# @jit(nopython=True)
# def convert_x_to_bbox(x_state):
#     """Convert state vector to bounding box."""
#     s_val = max(x_state[2], 1e-6)  # Prevent negative area
#     r_val = max(x_state[3], 1e-6)  # Prevent negative aspect ratio
    
#     w = math.sqrt(s_val * r_val)
#     h = s_val / w if w > 1e-6 else math.sqrt(s_val / (r_val + 1e-6))
#     h = max(h, 1e-6)  # Ensure h is not zero
    
#     return np.array([
#         x_state[0]-w/2.,
#         x_state[1]-h/2.,
#         x_state[0]+w/2.,
#         x_state[1]+h/2.
#     ]).reshape(1, 4)  # Return as 2D array with shape (1, 4)

# @jit(nopython=True)
# def calculate_adaptive_radius(base_radius, approach_speed, circle_radius, 
#                             speed_sensitivity, size_sensitivity,
#                             min_radius, max_radius):
#     """Calculate adaptive click radius based on speed and circle size."""
#     speed_bonus = approach_speed * speed_sensitivity
#     size_bonus = circle_radius * size_sensitivity
#     effective_radius = base_radius + speed_bonus + size_bonus
#     return max(min_radius, min(max_radius, effective_radius))

# # --- Association Logic (standard SORT) ---
# def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
#     """Associate detections to existing trackers using Hungarian algorithm."""
#     if len(trackers) == 0:
#         return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)

#     iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    
#     for d, det in enumerate(detections):
#         for t, trk in enumerate(trackers):
#             iou_matrix[d,t] = iou(det, trk)

#     # Hungarian algorithm for assignment
#     row_ind, col_ind = linear_sum_assignment(-iou_matrix)
#     matched_indices = np.column_stack((row_ind, col_ind))

#     unmatched_detections = []
#     for d, det in enumerate(detections):
#         if d not in matched_indices[:,0]:
#             unmatched_detections.append(d)
    
#     unmatched_trackers = []
#     for t, trk in enumerate(trackers):
#         if t not in matched_indices[:,1]:
#             unmatched_trackers.append(t)

#     matches = []
#     for m in matched_indices:
#         if iou_matrix[m[0],m[1]] < iou_threshold:
#             unmatched_detections.append(m[0])
#             unmatched_trackers.append(m[1])
#         else:
#             matches.append(m.reshape(1,2))
    
#     matches = np.concatenate(matches, axis=0) if len(matches) > 0 else np.empty((0,2), dtype=int)
#     return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# # --- Mouse Control ---
# def click(x, y, hold=False):
#     """Perform mouse click at specified coordinates."""
#     win32api.SetCursorPos((int(x), int(y)))
#     if hold:
#         win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
#     else:
#         win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
#         time.sleep(0.001)
#         win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

# def release_click():
#     """Release mouse click."""
#     win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

# # --- Slider Detection (Heuristic) ---
# def is_slider_ball(track_id, history, min_history_len=5, min_avg_movement=2.0, max_movement_std=5.0):
#     """Detect if a track is likely a slider ball based on movement patterns."""
#     if track_id not in history or len(history[track_id]) < min_history_len:
#         return False
    
#     recent_history = list(history[track_id])[-min_history_len:]
#     movements = []
    
#     for i in range(1, len(recent_history)):
#         prev_center, _ = recent_history[i-1]
#         curr_center, _ = recent_history[i]
#         movement = calculate_movement(prev_center, curr_center)
#         movements.append(movement)
    
#     if not movements:
#         return False
        
#     avg_movement = sum(movements) / len(movements)
#     movement_std = np.std(movements) if len(movements) > 1 else 0
    
#     return avg_movement > min_avg_movement and movement_std < max_movement_std

# # --- Main Tracker Logic ---
# class ClickPredictor:
#     def __init__(self):
#         self.last_click_time = 0
#         self.click_cooldown = CLICK_COOLDOWN
#         self.prediction_frames = PREDICTION_FRAMES
#         self.min_confidence = MIN_CONFIDENCE_FOR_CLICK
#         self.speed_weight = SPEED_WEIGHT
#         self.distance_weight = DISTANCE_WEIGHT
#         self.max_prediction_error = MAX_PREDICTION_ERROR
#         self.min_speed_for_prediction = 2  # Lowered from 5 to 2
#         self.base_radius = 20  # Increased from 15 to 20
#         self.speed_multiplier = 0.8  # Increased from 0.5 to 0.8
#         self.size_multiplier = 0.5  # Increased from 0.3 to 0.5

#     def predict_position(self, track_history, track_id):
#         if track_id not in track_history or len(track_history[track_id]) < 2:
#             return None, 0

#         history = list(track_history[track_id])
#         if len(history) < 2:
#             return None, 0

#         # Calculate velocity from recent history
#         recent_positions = [pos for pos, _ in history[-3:]]
#         if len(recent_positions) < 2:
#             return None, 0

#         velocities = []
#         for i in range(1, len(recent_positions)):
#             prev_x, prev_y = recent_positions[i-1]
#             curr_x, curr_y = recent_positions[i]
#             vx = curr_x - prev_x
#             vy = curr_y - prev_y
#             velocities.append((vx, vy))

#         # Average velocity
#         avg_vx = sum(v[0] for v in velocities) / len(velocities)
#         avg_vy = sum(v[1] for v in velocities) / len(velocities)
#         speed = math.sqrt(avg_vx**2 + avg_vy**2)

#         # Predict future position
#         last_pos = recent_positions[-1]
#         pred_x = last_pos[0] + avg_vx * self.prediction_frames
#         pred_y = last_pos[1] + avg_vy * self.prediction_frames

#         return (pred_x, pred_y), speed

#     def should_click(self, track_id, current_pos, track_history, is_slider=False):
#         current_time = time.time()
#         if current_time - self.last_click_time < self.click_cooldown:
#             return False, None

#         if track_id not in track_history or len(track_history[track_id]) < 2:
#             return False, None

#         prediction, speed = self.predict_position(track_history, track_id)
#         if prediction is None:
#             return False, None

#         pred_x, pred_y = prediction
#         curr_x, curr_y = current_pos

#         # Calculate distance to prediction
#         distance = math.sqrt((pred_x - curr_x)**2 + (pred_y - curr_y)**2)
        
#         # Calculate adaptive click radius based on speed and object size
#         click_radius = calculate_adaptive_radius(
#             self.base_radius,
#             speed,
#             distance,
#             self.speed_multiplier,
#             self.size_multiplier,
#             self.base_radius,
#             self.base_radius * 2
#         )

#         # For sliders, we want to follow them more closely
#         if is_slider:
#             click_radius *= 1.5

#         # Determine if we should click based on distance and speed
#         should_click = distance <= click_radius and speed >= self.min_speed_for_prediction

#         # Debug print
#         if track_id % 10 == 0:  # Print every 10th frame to avoid spam
#             print(f"Track {track_id} - Distance: {distance:.1f}, Speed: {speed:.1f}, "
#                   f"Click Radius: {click_radius:.1f}, Should Click: {should_click}")

#         if should_click:
#             self.last_click_time = current_time

#         return should_click, click_radius

# class CircleTracker:
#     def __init__(self, max_age, min_hits, iou_threshold, screen_height_for_thresholding):
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.trackers = []
#         self.frame_count = 0
#         self.track_history = defaultdict(lambda: deque(maxlen=10))
#         self.active_sliders = set()
#         self.clicked_circles = set()
#         self.click_predictor = ClickPredictor()
#         self.screen_height_for_thresholding = screen_height_for_thresholding
#         self.current_slider = None  # Track current slider being followed

#     def process_click(self, track_id, screen_x, screen_y, is_slider=False):
#         """Process a click event."""
#         if is_slider:
#             if track_id not in self.active_sliders:
#                 self.active_sliders.add(track_id)
#                 click(screen_x, screen_y, hold=True)
#                 print(f"Started following slider {track_id} at ({screen_x:.0f}, {screen_y:.0f})")
#             else:
#                 # Update cursor position for slider
#                 win32api.SetCursorPos((int(screen_x), int(screen_y)))
#         else:
#             self.clicked_circles.add(track_id)
#             click(screen_x, screen_y)
#             print(f"Clicked circle {track_id} at ({screen_x:.0f}, {screen_y:.0f})")

#     def update(self, detections_xyxyscore):
#         """Update trackers with new detections."""
#         self.frame_count += 1
        
#         # Get predicted locations from existing trackers
#         trks = np.zeros((len(self.trackers), 5))
#         to_del = []
#         ret = []
        
#         for t, trk_obj in enumerate(self.trackers):
#             pos = trk_obj.predict()[0]
#             trks[t] = [pos[0], pos[1], pos[2], pos[3], trk_obj.id]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
        
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in sorted(to_del, reverse=True):
#             self.trackers.pop(t)

#         matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
#             detections_xyxyscore[:, :4], trks[:, :4], self.iou_threshold)

#         # Update matched trackers
#         for m in matched:
#             det_idx, trk_idx = m[0], m[1]
#             self.trackers[trk_idx].update(detections_xyxyscore[det_idx, :4])

#         # Create new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(detections_xyxyscore[i, :4])
#             self.trackers.append(trk)

#         # Update track states and remove old tracks
#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             d = trk.get_state()[0]
#             if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
#                 ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
#             i -= 1
#             if(trk.time_since_update > self.max_age):
#                 if trk.id in self.clicked_circles:
#                     self.clicked_circles.remove(trk.id)
#                 if trk.id in self.active_sliders:
#                     self.active_sliders.remove(trk.id)
#                     release_click()  # Release click when slider track is lost
#                 if trk.id in self.track_history:
#                     del self.track_history[trk.id]
#                 self.trackers.pop(i)

#         return np.vstack(ret) if len(ret) > 0 else np.empty((0, 5))

# class FPSCounter:
#     def __init__(self, window_size=60):
#         self.window_size = window_size
#         self.frame_times = deque(maxlen=window_size)
#         self.last_time = time.perf_counter()
#         self.frame_count = 0
#         self.fps = 0
#         self.min_fps = float('inf')
#         self.max_fps = 0
#         self.avg_frame_time_ms = 0
#         self.last_print_time = time.perf_counter()
#         self.print_interval = SHOW_FPS_STATS_INTERVAL

#     def update(self):
#         current_time = time.perf_counter()
#         frame_time = current_time - self.last_time
#         self.frame_times.append(frame_time)
#         self.last_time = current_time
#         self.frame_count += 1

#         if len(self.frame_times) > 0:
#             avg_interval = statistics.mean(self.frame_times)
#             self.fps = 1.0 / avg_interval if avg_interval > 0 else 0
#             self.min_fps = min(self.min_fps, 1.0 / max(self.frame_times) if max(self.frame_times) > 0 else 0)
#             self.max_fps = max(self.max_fps, 1.0 / min(self.frame_times) if min(self.frame_times) > 0 else 0)
#             self.avg_frame_time_ms = avg_interval * 1000

#         if current_time - self.last_print_time >= self.print_interval:
#             self.print_stats()
#             self.last_print_time = current_time
#             # Reset min/max for the next interval to make them more relevant to current performance
#             self.min_fps = self.fps 
#             self.max_fps = self.fps

#     def print_stats(self):
#         print(f"FPS: {self.fps:.1f} (Min: {self.min_fps:.1f}, Max: {self.max_fps:.1f}) | Avg Frame Time: {self.avg_frame_time_ms:.1f}ms | Frames: {self.frame_count}")

# # --- Threading Functions ---
# STOP_EVENT = threading.Event()

# def capture_thread_func(camera, capture_queue, target_fps):
#     """Thread function for screen capture."""
#     print("Capture thread started.")
#     try:
#         camera.start(target_fps=target_fps, video_mode=False)
#         last_frame_time = time.perf_counter()
#         frame_interval = 1.0 / target_fps
        
#         while not STOP_EVENT.is_set():
#             try:
#                 current_time = time.perf_counter()
#                 if current_time - last_frame_time >= frame_interval:
#                     frame = camera.get_latest_frame()
#                     if frame is not None:
#                         try:
#                             capture_queue.put(frame, block=False, timeout=frame_interval)
#                             last_frame_time = current_time
#                         except:
#                             pass  # Queue is full, skip this frame
#                     else:
#                         time.sleep(frame_interval * 0.1)  # Small sleep if no frame
#                 else:
#                     time.sleep(frame_interval * 0.1)  # Sleep until next frame time
#             except Exception as e:
#                 print(f"Error in capture loop: {e}")
#                 time.sleep(0.1)  # Sleep on error
#                 continue
#     except Exception as e:
#         print(f"Fatal error in capture thread: {e}")
#     finally:
#         try:
#             camera.stop()
#         except:
#             pass
#         print("Capture thread stopped.")

# def inference_thread_func(model, capture_queue, result_queue, device):
#     """Thread function for running YOLO inference."""
#     print(f"Inference thread started on device: {device}.")
#     while not STOP_EVENT.is_set():
#         try:
#             frame = capture_queue.get(block=True, timeout=0.1) # Wait for a frame
#             # YOLO expects BGR, DXCam provides BGR by default
#             results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, imgsz=INFERENCE_SIZE,
#                                   verbose=False, device=device)
            
#             # Extract detections: [x1, y1, x2, y2, score]
#             if results and results[0].boxes is not None and len(results[0].boxes.data) > 0:
#                 # Ensure we have the correct shape [N, 5] where N is number of detections
#                 boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4]
#                 scores = results[0].boxes.conf.cpu().numpy()  # [N]
#                 detections = np.column_stack((boxes, scores))  # [N, 5]
#             else:
#                 detections = np.empty((0, 5))
            
#             try:
#                 result_queue.put((detections, frame.shape, frame), block=False, timeout=0.001)
#             except:
#                 pass # Result queue full, skip
#             capture_queue.task_done()
#         except Empty:
#             continue # Capture queue was empty
#     print("Inference thread stopped.")

# def draw_visualization(frame, tracked_objects, circle_tracker, frame_center_x, frame_center_y):
#     """Draw visualization of detections, tracking, and click radius."""
#     vis_frame = frame.copy()
    
#     # Draw center crosshair
#     cv2.line(vis_frame, (frame_center_x - 10, frame_center_y), (frame_center_x + 10, frame_center_y), (0, 255, 0), 2)
#     cv2.line(vis_frame, (frame_center_x, frame_center_y - 10), (frame_center_x, frame_center_y + 10), (0, 255, 0), 2)
    
#     # Draw tracked objects
#     if tracked_objects.size > 0:
#         for obj in tracked_objects:
#             if obj.size != 5:
#                 continue
                
#             x1, y1, x2, y2, track_id = obj
#             obj_center_x = int((x1 + x2) / 2)
#             obj_center_y = int((y1 + y2) / 2)
#             obj_radius = max(x2 - x1, y2 - y1) / 2
            
#             # Determine if it's a slider
#             is_slider = is_slider_ball(track_id, circle_tracker.track_history)
            
#             # Get prediction and click decision
#             prediction, speed = circle_tracker.click_predictor.predict_position(circle_tracker.track_history, track_id)
#             should_click, click_radius = circle_tracker.click_predictor.should_click(
#                 track_id, (obj_center_x, obj_center_y), circle_tracker.track_history, is_slider
#             )
            
#             # Draw bounding box with different colors
#             if is_slider:
#                 color = (0, 255, 255) if track_id in circle_tracker.active_sliders else (255, 255, 0)  # Yellow/Cyan for sliders
#             else:
#                 color = (0, 255, 0) if track_id not in circle_tracker.clicked_circles else (0, 0, 255)  # Green/Red for hit circles
                
#             cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
#             # Draw click radius if we have a prediction
#             if prediction is not None and click_radius is not None:
#                 pred_x, pred_y = prediction
#                 cv2.circle(vis_frame, (obj_center_x, obj_center_y), int(click_radius), (255, 0, 0), 1)
#                 cv2.circle(vis_frame, (int(pred_x), int(pred_y)), 3, (0, 255, 255), -1)
#                 cv2.line(vis_frame, (obj_center_x, obj_center_y), (int(pred_x), int(pred_y)), (0, 255, 255), 1)
            
#             # Draw track ID and type
#             cv2.putText(vis_frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#             cv2.putText(vis_frame, "Slider" if is_slider else "Hit", (int(x1), int(y1) - 30),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#             if prediction is not None:
#                 cv2.putText(vis_frame, f"Speed: {speed:.1f}", (int(x1), int(y1) - 50),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     # Draw FPS
#     cv2.putText(vis_frame, f"FPS: {fps_counter.fps:.1f}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
#     return vis_frame

# # --- Main Application ---
# if __name__ == "__main__":
#     # Initialize YOLO model
#     device = 'cuda:0' if USE_GPU and torch.cuda.is_available() else 'cpu'
#     try:
#         # Load ONNX model
#         model = YOLO(YOLO_MODEL_PATH)
#         print(f"YOLO model loaded on {device}.")
#     except Exception as e:
#         print(f"Error loading YOLO model: {e}")
#         exit()

#     # Initialize screen capture
#     camera = None
#     try:
#         camera = dxcam.create(output_idx=0, output_color="BGR")
#         # Test capture one frame to get dimensions
#         test_frame = camera.grab()
#         if test_frame is None:
#             raise Exception("DXCam could not grab an initial frame.")
#         frame_height, frame_width = test_frame.shape[:2]
#         print(f"Screen capture initialized: {frame_width}x{frame_height}")
#     except Exception as e:
#         print(f"Error initializing screen capture: {e}")
#         if camera:
#             try:
#                 camera.stop()
#             except:
#                 pass
#         exit()

#     # Screen dimensions for coordinate mapping
#     screen_width_sys = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
#     screen_height_sys = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    
#     # Scaling factors (if captured frame size != screen size, though DXCam usually matches)
#     scale_x = screen_width_sys / frame_width
#     scale_y = screen_height_sys / frame_height

#     # Initialize Tracker
#     circle_tracker = CircleTracker(max_age=MAX_AGE, min_hits=MIN_HITS_TO_CONFIRM, 
#                                    iou_threshold=IOU_THRESHOLD, 
#                                    screen_height_for_thresholding=frame_height)
#     fps_counter = FPSCounter()

#     # Queues for threading
#     capture_q = Queue(maxsize=2)  # Small buffer to keep frames fresh
#     result_q = Queue(maxsize=2)   # Small buffer for results

#     # Start threads
#     cap_thread = threading.Thread(target=capture_thread_func, args=(camera, capture_q, DXCAM_FPS_TARGET), daemon=True)
#     inf_thread = threading.Thread(target=inference_thread_func, args=(model, capture_q, result_q, device), daemon=True)
    
#     cap_thread.start()
#     inf_thread.start()
    
#     print("Starting main loop...")
#     try:
#         while True:
#             try:
#                 detections_xyxyscore, frame_shape_from_inf, frame = result_q.get(block=True, timeout=0.1)
                
#                 tracked_objects = circle_tracker.update(detections_xyxyscore)

#                 frame_center_x = frame_width // 2
#                 frame_center_y = frame_height // 2

#                 # Process each tracked object
#                 if tracked_objects.size > 0:
#                     for obj in tracked_objects:
#                         if obj.size != 5:
#                             continue
                            
#                         x1, y1, x2, y2, track_id = obj
#                         obj_center_x = (x1 + x2) / 2
#                         obj_center_y = (y1 + y2) / 2
#                         obj_radius = max(x2 - x1, y2 - y1) / 2

#                         # Check if it's a slider
#                         is_slider = is_slider_ball(track_id, circle_tracker.track_history)

#                         # Get prediction and click decision
#                         should_click, click_radius = circle_tracker.click_predictor.should_click(
#                             track_id, (obj_center_x, obj_center_y), circle_tracker.track_history, is_slider
#                         )

#                         if should_click:
#                             screen_click_x = obj_center_x * scale_x
#                             screen_click_y = obj_center_y * scale_y
#                             circle_tracker.process_click(track_id, screen_click_x, screen_click_y, is_slider)

#                 vis_frame = draw_visualization(frame, tracked_objects, circle_tracker, frame_center_x, frame_center_y)
#                 vis_frame = cv2.resize(vis_frame, (0, 0), fx=0.5, fy=0.5)
#                 cv2.imshow('YOLO Detection and Tracking', vis_frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
                
#                 result_q.task_done()
#                 fps_counter.update()

#             except Empty:
#                 time.sleep(0.001)
#             except Exception as e:
#                 print(f"Error in main loop: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 time.sleep(0.1)

#     except KeyboardInterrupt:
#         print("\nCtrl+C received. Stopping...")
#     finally:
#         STOP_EVENT.set()
#         try:
#             if camera:
#                 camera.stop()
#         except:
#             pass
        
#         # Release any held clicks
#         release_click()
        
#         if 'cap_thread' in locals() and cap_thread.is_alive():
#             cap_thread.join(timeout=2)
#         if 'inf_thread' in locals() and inf_thread.is_alive():
#             inf_thread.join(timeout=2)
        
#         cv2.destroyAllWindows()
#         print("Program terminated.")
#         if 'fps_counter' in locals():
#             fps_counter.print_stats()