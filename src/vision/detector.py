import json
import cv2
import numpy as np
from ultralytics import YOLO

class AirportDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the YOLOv8 model for airport-specific object detection.
        Optimized for edge-based processing (Jetson Nano/Edge devices).
        
        :param model_path: Path to the YOLOv8 model weights (.pt or .onnx).
        """
        # Load model. Using 'yolov8n.pt' (Nano) by default for edge compatibility.
        self.model = YOLO(model_path)
        
        # Mapping COCO class IDs to SmartFlow Airport specific categories.
        # 0: person, 24: backpack, 26: handbag (Carry-on), 28: suitcase (Check-in luggage)
        self.class_mapping = {
            0: "Person",
            24: "Backpack",
            26: "Carry-on",
            28: "Check-in luggage"
        }
        self.target_classes = list(self.class_mapping.keys())

    def process_frame(self, frame):
        """
        Runs inference on a single frame and returns lightweight detection results.
        
        :param frame: OpenCV image (numpy array).
        :return: List of detections (Dictionary format).
        """
        # Run inference only on target classes
        results = self.model(frame, classes=self.target_classes, verbose=False)
        
        detections = []
        
        # Parse results into a lightweight JSON-compatible structure
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # [x1, y1, x2, y2]
                bbox = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                detections.append({
                    "bbox": [round(coord, 2) for coord in bbox],
                    "class_id": cls_id,
                    "class_name": self.class_mapping.get(cls_id, "Unknown"),
                    "confidence": round(conf, 4)
                })
        
        return detections

if __name__ == "__main__":
    # Local verification
    detector = AirportDetector()
    # Create a blank image for testing
    test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    output = detector.process_frame(test_frame)
    print("Lightweight Detections Output:")
    print(json.dumps(output, indent=2))
