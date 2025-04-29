# import os
# import cv2
# import numpy as np
# import torch
# import argparse
# import time
# from pathlib import Path
# from collections import defaultdict, deque

# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors

# class ObjectChangeDetector:
#     def __init__(self, model_path='yolov8n.pt', conf_thresh=0.3, iou_thresh=0.5, 
#                  device='', memory_frames=30, detection_threshold=15, 
#                  stable_frames=10, debug=False):
#         """
#         Initialize the Object Change Detector
        
#         Args:
#             model_path: Path to the YOLOv8 model
#             conf_thresh: Confidence threshold for detections
#             iou_thresh: IOU threshold for NMS
#             device: Device to run inference on ('cuda', 'cpu', etc.)
#             memory_frames: Number of frames to keep in memory for object history
#             detection_threshold: Number of frames an object needs to be present to be considered stable
#             stable_frames: Number of frames an object needs to be missing to be considered removed
#             debug: Enable debug mode for additional outputs
#         """
#         # Load the model
#         self.model = YOLO(model_path)
#         self.conf_thresh = conf_thresh
#         self.iou_thresh = iou_thresh
#         self.device = device
        
#         # Object tracking parameters
#         self.memory_frames = memory_frames
#         self.detection_threshold = detection_threshold
#         self.stable_frames = stable_frames
#         self.debug = debug
        
#         # Object tracking data structures
#         self.object_history = defaultdict(lambda: deque(maxlen=self.memory_frames))
#         self.stable_objects = {}  # Objects that have been consistently detected
#         self.missing_objects = {}  # Objects that were stable but are now missing
#         self.new_objects = {}  # Objects that have newly appeared
#         self.frame_counter = 0
        
#         # Performance metrics
#         self.processing_times = deque(maxlen=30)  # Store last 30 frames processing time
    
#     def process_frame(self, frame):
#         """
#         Process a single frame to detect missing and new objects
        
#         Args:
#             frame: Input frame (numpy array in BGR format)
            
#         Returns:
#             Annotated frame, FPS
#         """
#         start_time = time.time()
        
#         # Increment frame counter
#         self.frame_counter += 1
        
#         # Run object detection
#         results = self.model.track(frame, persist=True, verbose=False, conf=self.conf_thresh, 
#                                    iou=self.iou_thresh, device=self.device)
        
#         # Extract detections with tracking IDs
#         boxes = results[0].boxes
        
#         # Create annotator
#         annotator = Annotator(frame.copy(), line_width=2)
        
#         # Process detections with tracking IDs
#         current_objects = {}
#         if hasattr(boxes, 'id') and boxes.id is not None:
#             detected_ids = boxes.id.int().cpu().tolist()
#             detected_boxes = boxes.xyxy.cpu().numpy()
#             detected_confs = boxes.conf.cpu().numpy()
#             detected_cls = boxes.cls.cpu().numpy()
            
#             # Update tracking histories
#             for i, track_id in enumerate(detected_ids):
#                 box = detected_boxes[i]
#                 cls_id = int(detected_cls[i])
#                 conf = float(detected_confs[i])
#                 cls_name = self.model.names[cls_id]
                
#                 # Create a unique object ID
#                 obj_id = f"{cls_name}_{track_id}"
                
#                 # Store current detections
#                 current_objects[obj_id] = {
#                     "box": box,
#                     "class": cls_name,
#                     "confidence": conf,
#                     "track_id": track_id,
#                     "last_seen": self.frame_counter
#                 }
                
#                 # Update object history
#                 self.object_history[obj_id].append(1)  # Mark as detected in this frame
                
#                 # Check if object should be considered stable
#                 if (obj_id not in self.stable_objects and 
#                     sum(self.object_history[obj_id]) >= self.detection_threshold):
#                     self.stable_objects[obj_id] = current_objects[obj_id]
                    
#                     # If this was previously missing, remove from missing
#                     if obj_id in self.missing_objects:
#                         del self.missing_objects[obj_id]
                    
#                     # Mark as new object if it wasn't previously tracked
#                     if obj_id not in self.new_objects and self.frame_counter > self.memory_frames:
#                         self.new_objects[obj_id] = {
#                             **current_objects[obj_id],
#                             "detected_at": self.frame_counter
#                         }
        
#         # Update tracking status for objects not detected in current frame
#         for obj_id in list(self.object_history.keys()):
#             if obj_id not in current_objects:
#                 self.object_history[obj_id].append(0)  # Mark as not detected
                
#                 # Check if object should be considered missing
#                 if (obj_id in self.stable_objects and 
#                     sum(self.object_history[obj_id]) <= (self.memory_frames - self.stable_frames)):
#                     # Object was stable but now consistently missing
#                     if obj_id not in self.missing_objects:
#                         self.missing_objects[obj_id] = {
#                             **self.stable_objects[obj_id],
#                             "missing_since": self.frame_counter
#                         }
        
#         # Draw annotations
#         for obj_id, obj_data in current_objects.items():
#             box = obj_data["box"]
#             cls_name = obj_data["class"]
#             conf = obj_data["confidence"]
#             track_id = obj_data["track_id"]
            
#             # Determine annotation color based on object status
#             if obj_id in self.new_objects and self.frame_counter - self.new_objects[obj_id]["detected_at"] < 30:
#                 # New object (green)
#                 color = (0, 255, 0)
#                 label = f"NEW: {cls_name} {track_id} {conf:.2f}"
#             elif obj_id in self.stable_objects:
#                 # Stable object (blue)
#                 color = (255, 0, 0)
#                 label = f"{cls_name} {track_id} {conf:.2f}"
#             else:
#                 # Regular detection (white)
#                 color = colors(track_id)
#                 label = f"{cls_name} {track_id} {conf:.2f}"
            
#             annotator.box_label(box, label, color=color)
        
#         # Draw missing objects (last known position)
#         for obj_id, obj_data in self.missing_objects.items():
#             # Only show recently missing objects (within last 60 frames)
#             if self.frame_counter - obj_data["missing_since"] < 60:
#                 box = obj_data["box"]
#                 cls_name = obj_data["class"]
#                 track_id = obj_data["track_id"]
                
#                 # Red color for missing objects
#                 annotator.box_label(
#                     box, 
#                     f"MISSING: {cls_name} {track_id}", 
#                     color=(0, 0, 255)
#                 )
        
#         # Add status information
#         output_frame = annotator.result()
#         cv2.putText(output_frame, f"Tracking: {len(current_objects)} objects", 
#                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#         cv2.putText(output_frame, f"Stable: {len(self.stable_objects)}", 
#                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#         cv2.putText(output_frame, f"New: {len(self.new_objects)}", 
#                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         cv2.putText(output_frame, f"Missing: {len(self.missing_objects)}", 
#                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
#         # Calculate processing time and FPS
#         end_time = time.time()
#         processing_time = end_time - start_time
#         self.processing_times.append(processing_time)
        
#         avg_time = sum(self.processing_times) / len(self.processing_times)
#         fps = 1.0 / avg_time if avg_time > 0 else 0
        
#         cv2.putText(output_frame, f"FPS: {fps:.1f}", 
#                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
#         return output_frame, fps
    
#     def get_statistics(self):
#         """Get detector statistics"""
#         return {
#             "total_objects": len(self.object_history),
#             "stable_objects": len(self.stable_objects),
#             "missing_objects": len(self.missing_objects),
#             "new_objects": len(self.new_objects),
#             "frame_count": self.frame_counter,
#             "avg_fps": 1.0 / (sum(self.processing_times) / len(self.processing_times)) if self.processing_times else 0
#         }

# def main():
#     parser = argparse.ArgumentParser(description="Object Change Detection in Video")
#     parser.add_argument('--source', type=str, default='0', help='Source video file or webcam index')
#     parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
#     parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
#     parser.add_argument('--iou', type=float, default=0.5, help='IOU threshold for NMS')
#     parser.add_argument('--device', type=str, default='', help='Inference device (cuda/cpu)')
#     parser.add_argument('--output', type=str, default='output.mp4', help='Output video path')
#     parser.add_argument('--memory', type=int, default=30, help='Memory frames for tracking')
#     parser.add_argument('--stable-threshold', type=int, default=15, 
#                         help='Frames needed for an object to be considered stable')
#     parser.add_argument('--missing-threshold', type=int, default=10, 
#                         help='Frames needed for a stable object to be considered missing')
#     parser.add_argument('--no-save', action='store_true', help='Do not save output video')
#     parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
#     args = parser.parse_args()
    
#     # Initialize the detector
#     detector = ObjectChangeDetector(
#         model_path=args.model,
#         conf_thresh=args.conf,
#         iou_thresh=args.iou,
#         device=args.device,
#         memory_frames=args.memory,
#         detection_threshold=args.stable_threshold,
#         stable_frames=args.missing_threshold,
#         debug=args.debug
#     )
    
#     # Set up video source
#     if args.source.isdigit():
#         cap = cv2.VideoCapture(int(args.source))
#     else:
#         cap = cv2.VideoCapture(args.source)
    
#     if not cap.isOpened():
#         print(f"Error: Could not open video source {args.source}")
#         return
    
#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # Set up video writer if saving is enabled
#     writer = None
#     if not args.no_save:
#         output_path = args.output
#         writer = cv2.VideoWriter(
#             output_path, 
#             cv2.VideoWriter_fourcc(*'mp4v'), 
#             fps, 
#             (width, height)
#         )
    
#     # Process video
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Process the frame
#             output_frame, current_fps = detector.process_frame(frame)
            
#             # Display the frame
#             cv2.imshow('Object Change Detection', output_frame)
            
#             # Write frame to output video if enabled
#             if writer is not None:
#                 writer.write(output_frame)
            
#             # Break on 'q' key
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
    
#     finally:
#         # Clean up
#         cap.release()
#         if writer is not None:
#             writer.release()
#         cv2.destroyAllWindows()
        
#         # Print final statistics
#         stats = detector.get_statistics()
#         print("\nFinal Statistics:")
#         print(f"Total frames processed: {stats['frame_count']}")
#         print(f"Average FPS: {stats['avg_fps']:.2f}")
#         print(f"Total tracked objects: {stats['total_objects']}")
#         print(f"Stable objects: {stats['stable_objects']}")
#         print(f"Missing objects: {stats['missing_objects']}")
#         print(f"New objects: {stats['new_objects']}")

# if __name__ == "__main__":
#     main()


import os
import cv2
import numpy as np
import torch
import argparse
import time
from pathlib import Path
from collections import defaultdict, deque

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class ObjectChangeDetector:
    # The class remains unchanged
    # ... [your existing ObjectChangeDetector class] ...
    def __init__(self, model_path='yolov8n.pt', conf_thresh=0.3, iou_thresh=0.5, 
                 device='', memory_frames=30, detection_threshold=15, 
                 stable_frames=10, debug=False):
        """
        Initialize the Object Change Detector
        
        Args:
            model_path: Path to the YOLOv8 model
            conf_thresh: Confidence threshold for detections
            iou_thresh: IOU threshold for NMS
            device: Device to run inference on ('cuda', 'cpu', etc.)
            memory_frames: Number of frames to keep in memory for object history
            detection_threshold: Number of frames an object needs to be present to be considered stable
            stable_frames: Number of frames an object needs to be missing to be considered removed
            debug: Enable debug mode for additional outputs
        """
        # Load the model
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        
        # Object tracking parameters
        self.memory_frames = memory_frames
        self.detection_threshold = detection_threshold
        self.stable_frames = stable_frames
        self.debug = debug
        
        # Object tracking data structures
        self.object_history = defaultdict(lambda: deque(maxlen=self.memory_frames))
        self.stable_objects = {}  # Objects that have been consistently detected
        self.missing_objects = {}  # Objects that were stable but are now missing
        self.new_objects = {}  # Objects that have newly appeared
        self.frame_counter = 0
        
        # Performance metrics
        self.processing_times = deque(maxlen=30)  # Store last 30 frames processing time
    
    def process_frame(self, frame):
        """
        Process a single frame to detect missing and new objects
        
        Args:
            frame: Input frame (numpy array in BGR format)
            
        Returns:
            Annotated frame, FPS
        """
        start_time = time.time()
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Run object detection
        results = self.model.track(frame, persist=True, verbose=False, conf=self.conf_thresh, 
                                   iou=self.iou_thresh, device=self.device)
        
        # Extract detections with tracking IDs
        boxes = results[0].boxes
        
        # Create annotator
        annotator = Annotator(frame.copy(), line_width=2)
        
        # Process detections with tracking IDs
        current_objects = {}
        if hasattr(boxes, 'id') and boxes.id is not None:
            detected_ids = boxes.id.int().cpu().tolist()
            detected_boxes = boxes.xyxy.cpu().numpy()
            detected_confs = boxes.conf.cpu().numpy()
            detected_cls = boxes.cls.cpu().numpy()
            
            # Update tracking histories
            for i, track_id in enumerate(detected_ids):
                box = detected_boxes[i]
                cls_id = int(detected_cls[i])
                conf = float(detected_confs[i])
                cls_name = self.model.names[cls_id]
                
                # Create a unique object ID
                obj_id = f"{cls_name}_{track_id}"
                
                # Store current detections
                current_objects[obj_id] = {
                    "box": box,
                    "class": cls_name,
                    "confidence": conf,
                    "track_id": track_id,
                    "last_seen": self.frame_counter
                }
                
                # Update object history
                self.object_history[obj_id].append(1)  # Mark as detected in this frame
                
                # Check if object should be considered stable
                if (obj_id not in self.stable_objects and 
                    sum(self.object_history[obj_id]) >= self.detection_threshold):
                    self.stable_objects[obj_id] = current_objects[obj_id]
                    
                    # If this was previously missing, remove from missing
                    if obj_id in self.missing_objects:
                        del self.missing_objects[obj_id]
                    
                    # Mark as new object if it wasn't previously tracked
                    if obj_id not in self.new_objects and self.frame_counter > self.memory_frames:
                        self.new_objects[obj_id] = {
                            **current_objects[obj_id],
                            "detected_at": self.frame_counter
                        }
        
        # Update tracking status for objects not detected in current frame
        for obj_id in list(self.object_history.keys()):
            if obj_id not in current_objects:
                self.object_history[obj_id].append(0)  # Mark as not detected
                
                # Check if object should be considered missing
                if (obj_id in self.stable_objects and 
                    sum(self.object_history[obj_id]) <= (self.memory_frames - self.stable_frames)):
                    # Object was stable but now consistently missing
                    if obj_id not in self.missing_objects:
                        self.missing_objects[obj_id] = {
                            **self.stable_objects[obj_id],
                            "missing_since": self.frame_counter
                        }
        
        # Draw annotations
        for obj_id, obj_data in current_objects.items():
            box = obj_data["box"]
            cls_name = obj_data["class"]
            conf = obj_data["confidence"]
            track_id = obj_data["track_id"]
            
            # Determine annotation color based on object status
            if obj_id in self.new_objects and self.frame_counter - self.new_objects[obj_id]["detected_at"] < 30:
                # New object (green)
                color = (0, 255, 0)
                label = f"NEW: {cls_name} {track_id} {conf:.2f}"
            elif obj_id in self.stable_objects:
                # Stable object (blue)
                color = (255, 0, 0)
                label = f"{cls_name} {track_id} {conf:.2f}"
            else:
                # Regular detection (white)
                color = colors(track_id)
                label = f"{cls_name} {track_id} {conf:.2f}"
            
            annotator.box_label(box, label, color=color)
        
        # Draw missing objects (last known position)
        for obj_id, obj_data in self.missing_objects.items():
            # Only show recently missing objects (within last 60 frames)
            if self.frame_counter - obj_data["missing_since"] < 60:
                box = obj_data["box"]
                cls_name = obj_data["class"]
                track_id = obj_data["track_id"]
                
                # Red color for missing objects
                annotator.box_label(
                    box, 
                    f"MISSING: {cls_name} {track_id}", 
                    color=(0, 0, 255)
                )
        
        # Add status information
        output_frame = annotator.result()
        cv2.putText(output_frame, f"Tracking: {len(current_objects)} objects", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Stable: {len(self.stable_objects)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(output_frame, f"New: {len(self.new_objects)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Missing: {len(self.missing_objects)}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Calculate processing time and FPS
        end_time = time.time()
        processing_time = end_time - start_time
        self.processing_times.append(processing_time)
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        cv2.putText(output_frame, f"FPS: {fps:.1f}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return output_frame, fps
    
    def get_statistics(self):
        """Get detector statistics"""
        return {
            "total_objects": len(self.object_history),
            "stable_objects": len(self.stable_objects),
            "missing_objects": len(self.missing_objects),
            "new_objects": len(self.new_objects),
            "frame_count": self.frame_counter,
            "avg_fps": 1.0 / (sum(self.processing_times) / len(self.processing_times)) if self.processing_times else 0
        }

def main():
    parser = argparse.ArgumentParser(description="Object Change Detection in Video")
    parser.add_argument('--source', type=str, default='0', help='Source video file or webcam index')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', type=str, default='', help='Inference device (cuda/cpu)')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video path')
    parser.add_argument('--memory', type=int, default=30, help='Memory frames for tracking')
    parser.add_argument('--stable-threshold', type=int, default=15, 
                        help='Frames needed for an object to be considered stable')
    parser.add_argument('--missing-threshold', type=int, default=10, 
                        help='Frames needed for a stable object to be considered missing')
    parser.add_argument('--no-save', action='store_true', help='Do not save output video')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no display)')
    
    args = parser.parse_args()
    
    # Initialize the detector
    detector = ObjectChangeDetector(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device=args.device,
        memory_frames=args.memory,
        detection_threshold=args.stable_threshold,
        stable_frames=args.missing_threshold,
        debug=args.debug
    )
    
    # Set up video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up video writer if saving is enabled
    writer = None
    if not args.no_save:
        output_path = args.output
        writer = cv2.VideoWriter(
            output_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (width, height)
        )
    
    # Process video
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}...")
            
            # Process the frame
            output_frame, current_fps = detector.process_frame(frame)
            
            # Display the frame if not in headless mode
            if not args.headless:
                cv2.imshow('Object Change Detection', output_frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write frame to output video if enabled
            if writer is not None:
                writer.write(output_frame)
    
    finally:
        # Clean up
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        
        # Print final statistics
        stats = detector.get_statistics()
        print("\nFinal Statistics:")
        print(f"Total frames processed: {stats['frame_count']}")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Total tracked objects: {stats['total_objects']}")
        print(f"Stable objects: {stats['stable_objects']}")
        print(f"Missing objects: {stats['missing_objects']}")
        print(f"New objects: {stats['new_objects']}")

if __name__ == "__main__":
    main()