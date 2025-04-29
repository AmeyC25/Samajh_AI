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
    def __init__(self, model_path='yolov8n.pt', conf_thresh=0.3, iou_thresh=0.5, 
                 device='', memory_frames=30, detection_threshold=15, 
                 stable_frames=10, debug=False):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device

        self.memory_frames = memory_frames
        self.detection_threshold = detection_threshold
        self.stable_frames = stable_frames
        self.debug = debug

        self.object_history = defaultdict(lambda: deque(maxlen=self.memory_frames))
        self.stable_objects = {}
        self.missing_objects = {}
        self.new_objects = {}
        self.frame_counter = 0
        self.processing_times = deque(maxlen=30)

    def process_frame(self, frame):
        start_time = time.time()
        self.frame_counter += 1

        results = self.model.track(frame, persist=True, verbose=False, conf=self.conf_thresh, 
                                   iou=self.iou_thresh, device=self.device)
        
        boxes = results[0].boxes
        annotator = Annotator(frame.copy(), line_width=2)

        current_objects = {}
        if hasattr(boxes, 'id') and boxes.id is not None:
            detected_ids = boxes.id.int().cpu().tolist()
            detected_boxes = boxes.xyxy.cpu().numpy()
            detected_confs = boxes.conf.cpu().numpy()
            detected_cls = boxes.cls.cpu().numpy()

            for i, track_id in enumerate(detected_ids):
                box = detected_boxes[i]
                cls_id = int(detected_cls[i])
                conf = float(detected_confs[i])
                cls_name = self.model.names[cls_id]

                obj_id = f"{cls_name}_{track_id}"

                current_objects[obj_id] = {
                    "box": box,
                    "class": cls_name,
                    "confidence": conf,
                    "track_id": track_id,
                    "last_seen": self.frame_counter
                }

                self.object_history[obj_id].append(1)

                if (obj_id not in self.stable_objects and 
                    sum(self.object_history[obj_id]) >= self.detection_threshold):
                    self.stable_objects[obj_id] = current_objects[obj_id]

                    if obj_id in self.missing_objects:
                        del self.missing_objects[obj_id]

                    if obj_id not in self.new_objects and self.frame_counter > self.memory_frames:
                        self.new_objects[obj_id] = {
                            **current_objects[obj_id],
                            "detected_at": self.frame_counter
                        }

        for obj_id in list(self.object_history.keys()):
            if obj_id not in current_objects:
                self.object_history[obj_id].append(0)

                if (obj_id in self.stable_objects and 
                    sum(self.object_history[obj_id]) <= (self.memory_frames - self.stable_frames)):
                    if obj_id not in self.missing_objects:
                        self.missing_objects[obj_id] = {
                            **self.stable_objects[obj_id],
                            "missing_since": self.frame_counter
                        }

        for obj_id, obj_data in current_objects.items():
            box = obj_data["box"]
            cls_name = obj_data["class"]
            conf = obj_data["confidence"]
            track_id = obj_data["track_id"]

            if obj_id in self.new_objects and self.frame_counter - self.new_objects[obj_id]["detected_at"] < 30:
                color = (0, 255, 0)
                label = f"NEW: {cls_name} {track_id} {conf:.2f}"
            elif obj_id in self.stable_objects:
                color = (255, 0, 0)
                label = f"{cls_name} {track_id} {conf:.2f}"
            else:
                color = colors(track_id)
                label = f"{cls_name} {track_id} {conf:.2f}"

            annotator.box_label(box, label, color=color)

        for obj_id, obj_data in self.missing_objects.items():
            if self.frame_counter - obj_data["missing_since"] < 60:
                box = obj_data["box"]
                cls_name = obj_data["class"]
                track_id = obj_data["track_id"]
                annotator.box_label(
                    box, f"MISSING: {cls_name} {track_id}", color=(0, 0, 255)
                )

        output_frame = annotator.result()

        # Stats overlay
        fps = 1.0 / (time.time() - start_time + 1e-6)
        self.processing_times.append(fps)
        avg_fps = sum(self.processing_times) / len(self.processing_times)

        cv2.putText(output_frame, f"Tracking: {len(current_objects)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output_frame, f"FPS: {avg_fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return output_frame

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=0, help='Video file or camera index')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--device', type=str, default='', help='Device to run on (e.g., "cpu" or "0")')
    parser.add_argument('--output', type=str, default=None, help='Path to save the output video')
    return parser.parse_args()


def main():
    args = parse_args()
    detector = ObjectChangeDetector(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device=args.device,
        debug=False
    )

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)

    assert cap.isOpened(), f"Failed to open source: {args.source}"

    # Get video writer if output path is provided
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = detector.process_frame(frame)
        if writer:
            writer.write(processed)

        cv2.imshow("Object Change Detector", processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
