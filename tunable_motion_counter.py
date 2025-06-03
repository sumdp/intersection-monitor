#!/usr/bin/env python3
"""
Tunable Motion Car Counter
Adjust sensitivity in real-time to eliminate false positives
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

class TunableMotionCounter:
    def __init__(self):
        """Initialize with adjustable motion detection"""
        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')

        # Tracking
        self.tracked_cars = {}
        self.next_car_id = 0

        # Counters
        self.moving_count = 0
        self.stationary_count = 0
        self.total_count = 0

        # Adjustable sensitivity settings
        self.sensitivity_presets = {
            'high': {'threshold': 2.0, 'min_distance': 15, 'stationary_frames': 10},
            'medium': {'threshold': 5.0, 'min_distance': 30, 'stationary_frames': 20},
            'low': {'threshold': 8.0, 'min_distance': 50, 'stationary_frames': 30},
            'very_low': {'threshold': 12.0, 'min_distance': 70, 'stationary_frames': 40}
        }
        self.current_sensitivity = 'medium'

        # Apply current sensitivity
        preset = self.sensitivity_presets[self.current_sensitivity]
        self.min_movement_threshold = preset['threshold']
        self.min_total_distance = preset['min_distance']
        self.stationary_frames_threshold = preset['stationary_frames']

        # Fixed settings
        self.max_distance = 100  # for tracking
        self.max_frames_missing = 5
        self.position_history_size = 20
        self.smoothing_window = 5

        # Vehicle classes
        self.vehicle_classes = [2, 3, 5, 7]

        # Debug mode
        self.debug_mode = False

    def apply_sensitivity(self, level):
        """Apply a sensitivity preset"""
        if level in self.sensitivity_presets:
            preset = self.sensitivity_presets[level]
            self.min_movement_threshold = preset['threshold']
            self.min_total_distance = preset['min_distance']
            self.stationary_frames_threshold = preset['stationary_frames']
            self.current_sensitivity = level
            print(f"Sensitivity changed to: {level.upper()}")
            print(f"  Movement threshold: {self.min_movement_threshold} px/frame")
            print(f"  Min total distance: {self.min_total_distance} px")
            print(f"  Stationary frames: {self.stationary_frames_threshold}")

    def find_closest_car(self, center, threshold=100):
        """Find the closest tracked car"""
        min_dist = float('inf')
        closest_id = None

        for car_id, info in self.tracked_cars.items():
            dist = np.linalg.norm(np.array(center) - np.array(info['center']))
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_id = car_id

        return closest_id

    def calculate_motion_metrics(self, positions):
        """Calculate comprehensive motion metrics"""
        if len(positions) < 3:
            return {'speed': 0, 'total_distance': 0, 'is_moving': False}

        positions_list = list(positions)

        # Calculate total distance
        total_distance = 0
        instant_speeds = []

        for i in range(1, len(positions_list)):
            dist = np.linalg.norm(np.array(positions_list[i]) - np.array(positions_list[i-1]))
            total_distance += dist
            instant_speeds.append(dist)

        # Calculate smoothed speed
        if len(positions_list) >= self.smoothing_window + 1:
            # Use positions that are several frames apart for stability
            smooth_speeds = []
            for i in range(self.smoothing_window, len(positions_list)):
                dist = np.linalg.norm(
                    np.array(positions_list[i]) -
                    np.array(positions_list[i-self.smoothing_window])
                ) / self.smoothing_window
                smooth_speeds.append(dist)

            # Use median for robustness against outliers
            avg_speed = np.median(smooth_speeds) if smooth_speeds else 0
        else:
            avg_speed = np.mean(instant_speeds) if instant_speeds else 0

        # Determine if moving based on multiple criteria
        is_moving = (
            avg_speed >= self.min_movement_threshold and
            total_distance >= self.min_total_distance
        )

        return {
            'speed': avg_speed,
            'total_distance': total_distance,
            'is_moving': is_moving,
            'instant_speeds': instant_speeds
        }

    def classify_motion(self, car_info):
        """Classify car motion with sensitivity settings"""
        metrics = self.calculate_motion_metrics(car_info['position_history'])

        car_info['speed'] = metrics['speed']
        car_info['total_distance'] = metrics['total_distance']

        if not metrics['is_moving']:
            car_info['stationary_frames'] += 1
        else:
            car_info['stationary_frames'] = 0

        # Determine status
        if car_info['stationary_frames'] >= self.stationary_frames_threshold:
            return 'stationary'
        elif metrics['is_moving']:
            return 'moving'
        else:
            return 'slowing'

    def process_frame(self, frame):
        """Process frame with tunable motion detection"""
        # Run detection
        results = self.model(frame, conf=0.55, verbose=False)

        current_cars = set()
        annotated = frame.copy()

        # Process detections
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                if cls not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Track vehicle
                car_id = self.find_closest_car(center)

                if car_id is None:
                    # New vehicle
                    car_id = self.next_car_id
                    self.next_car_id += 1
                    self.tracked_cars[car_id] = {
                        'center': center,
                        'position_history': deque(maxlen=self.position_history_size),
                        'first_seen': time.time(),
                        'frames_missing': 0,
                        'counted': False,
                        'status': 'new',
                        'speed': 0,
                        'total_distance': 0,
                        'stationary_frames': 0,
                        'bbox': (x1, y1, x2, y2)
                    }
                else:
                    # Update existing
                    self.tracked_cars[car_id]['center'] = center
                    self.tracked_cars[car_id]['frames_missing'] = 0
                    self.tracked_cars[car_id]['bbox'] = (x1, y1, x2, y2)

                # Update position history
                self.tracked_cars[car_id]['position_history'].append(center)

                # Classify motion
                if len(self.tracked_cars[car_id]['position_history']) >= 3:
                    self.tracked_cars[car_id]['status'] = self.classify_motion(self.tracked_cars[car_id])

                current_cars.add(car_id)

        # Draw tracked vehicles
        for car_id in current_cars:
            info = self.tracked_cars[car_id]
            x1, y1, x2, y2 = info['bbox']
            status = info['status']
            speed = info['speed']

            # Color based on status
            if status == 'moving':
                color = (0, 255, 0)  # Green
                thickness = 3
            elif status == 'stationary':
                color = (0, 0, 255)  # Red
                thickness = 2
            elif status == 'slowing':
                color = (0, 165, 255)  # Orange
                thickness = 2
            else:
                color = (255, 255, 0)  # Cyan
                thickness = 2

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Draw trail for moving vehicles
            if status == 'moving' and len(info['position_history']) > 3:
                points = np.array(list(info['position_history'])[-10:], dtype=np.int32)
                cv2.polylines(annotated, [points], False, color, 2)

            # Label
            label_parts = [f"ID:{car_id}"]
            if self.debug_mode:
                label_parts.extend([
                    f"S:{speed:.1f}",
                    f"D:{info['total_distance']:.0f}"
                ])
            elif status == 'moving':
                label_parts.append(f"{speed:.1f} px/f")

            label = " ".join(label_parts)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-20), (x1+label_size[0]+5, y1), color, -1)
            cv2.putText(annotated, label, (x1+2, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Handle cars that left
        to_remove = []
        for car_id, info in self.tracked_cars.items():
            if car_id not in current_cars:
                info['frames_missing'] += 1

                if info['frames_missing'] > self.max_frames_missing:
                    if not info['counted'] and time.time() - info['first_seen'] > 1.0:
                        self.total_count += 1
                        info['counted'] = True

                        if info['status'] == 'moving':
                            self.moving_count += 1
                            print(f"✅ Moving car {car_id} counted!")
                        else:
                            self.stationary_count += 1
                            print(f"🛑 Stationary car {car_id} counted!")

                    to_remove.append(car_id)

        for car_id in to_remove:
            del self.tracked_cars[car_id]

        # Draw UI
        self.draw_ui(annotated, frame.shape)

        return annotated

    def draw_ui(self, frame, shape):
        """Draw the user interface"""
        # Main panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 220), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Title and sensitivity
        cv2.putText(frame, f"Sensitivity: {self.current_sensitivity.upper()}",
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Counters
        y = 70
        cv2.putText(frame, f"Total Cars: {self.total_count}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 25
        cv2.putText(frame, f"Moving: {self.moving_count}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25
        cv2.putText(frame, f"Stationary: {self.stationary_count}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Settings display
        y += 30
        cv2.putText(frame, f"Threshold: {self.min_movement_threshold:.1f} px/f",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20
        cv2.putText(frame, f"Min Distance: {self.min_total_distance} px",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Controls
        y += 25
        cv2.putText(frame, "Controls: [1-4] Sensitivity [D]ebug [R]eset [Q]uit",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Sensitivity scale
        scale_y = 250
        cv2.putText(frame, "Sensitivity Levels:", (20, scale_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        scale_y += 20

        sensitivities = ['high', 'medium', 'low', 'very_low']
        for i, sens in enumerate(sensitivities):
            color = (255, 255, 0) if sens == self.current_sensitivity else (150, 150, 150)
            cv2.putText(frame, f"[{i+1}] {sens.upper()}",
                       (20 + i * 90, scale_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Debug info
        if self.debug_mode:
            cv2.putText(frame, "DEBUG MODE ON",
                       (shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def run(self):
        """Run the tunable counter"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n🚗 Tunable Motion Car Counter")
        print("\n📊 Adjust sensitivity to reduce false positives:")
        print("  [1] HIGH - Detects slight movements")
        print("  [2] MEDIUM - Balanced (default)")
        print("  [3] LOW - Only clear movements")
        print("  [4] VERY LOW - Only significant movements")
        print("\nOther controls:")
        print("  [D] Toggle debug mode")
        print("  [R] Reset counters")
        print("  [Q] Quit")
        print("-" * 50)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            annotated = self.process_frame(frame)

            # Display
            cv2.imshow('Tunable Motion Counter', annotated)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.moving_count = 0
                self.stationary_count = 0
                self.total_count = 0
                self.tracked_cars.clear()
                self.next_car_id = 0
                print("Counters reset!")
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('1'):
                self.apply_sensitivity('high')
            elif key == ord('2'):
                self.apply_sensitivity('medium')
            elif key == ord('3'):
                self.apply_sensitivity('low')
            elif key == ord('4'):
                self.apply_sensitivity('very_low')

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 Final Statistics:")
        print(f"Total: {self.total_count}")
        print(f"Moving: {self.moving_count}")
        print(f"Stationary: {self.stationary_count}")

if __name__ == "__main__":
    counter = TunableMotionCounter()
    counter.run()
