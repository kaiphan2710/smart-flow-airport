import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class AirportTracker:
    def __init__(self, max_age=30, n_init=3, nms_max_overlap=1.0):
        """
        Initializes the DeepSORT tracker for persistent person and luggage tracking.
        Optimized for edge computing using a lightweight mobilenet-based embedder.
        
        :param max_age: Maximum number of frames to keep a track alive without detection.
        :param n_init: Number of consecutive detections before a track is confirmed.
        """
        # Initialize the tracker with edge-optimized parameters
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            max_cosine_distance=0.3,
            nn_budget=None,
            embedder="mobilenet",  # Lightweight embedder for edge devices (Jetson Nano)
            half=True,             # FP16 precision
            bgr=True
        )

    def update(self, detections, frame):
        """
        Updates the tracker with detections from the detector module.
        
        :param detections: List of lightweight detection dictionaries from AirportDetector.
        :param frame: The current video frame (required for feature embedding).
        :return: List of tracked objects with IDs and associations.
        """
        # Convert detector output [x1, y1, x2, y2] to DeepSORT format [left, top, w, h]
        formatted_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            formatted_detections.append(([x1, y1, w, h], det['confidence'], det['class_id']))
        
        # Run DeepSORT update
        tracks = self.tracker.update_tracks(formatted_detections, frame=frame)
        
        tracked_objects = []
        for track in tracks:
            # Skip unconfirmed tracks (noise reduction)
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb() # Left, Top, Right, Bottom format
            class_id = track.get_det_class()
            
            tracked_objects.append({
                "track_id": track_id,
                "bbox": [round(coord, 2) for coord in ltrb],
                "class_id": class_id,
                "class_name": self._get_class_name(class_id)
            })
            
        # Perform Luggage-to-Person association
        return self._associate_luggage(tracked_objects)

    def _get_class_name(self, class_id):
        """Mirror the detector mapping for consistency."""
        mapping = {0: "Person", 24: "Backpack", 26: "Carry-on", 28: "Check-in luggage"}
        return mapping.get(class_id, "Unknown")

    def _associate_luggage(self, objects):
        """
        Logic for Edge Association:
        In a dense airport environment, we use spatial proximity (Euclidean distance between 
        bounding box centers) to associate luggage with the closest 'Person' track ID.
        
        This avoids the need for heavy ReID models for association, keeping it edge-compliant.
        """
        persons = [obj for obj in objects if obj['class_name'] == "Person"]
        luggages = [obj for obj in objects if obj['class_name'] != "Person"]
        
        for lug in luggages:
            lug_center = self._get_center(lug['bbox'])
            min_dist = float('inf')
            assigned_owner_id = None
            
            for person in persons:
                person_center = self._get_center(person['bbox'])
                dist = np.linalg.norm(np.array(lug_center) - np.array(person_center))
                
                # Check for closest proximity. 
                # Note: In production, we'd also check if the person is moving with the luggage.
                if dist < min_dist:
                    min_dist = dist
                    assigned_owner_id = person['track_id']
            
            # Spatial Threshold: Only associate if person is within a reasonable distance (e.g., 200px)
            if assigned_owner_id and min_dist < 200:
                lug['owner_id'] = assigned_owner_id
            else:
                lug['owner_id'] = "Unattended"

        return objects

    def _get_center(self, bbox):
        """Helper to find the center of a [x1, y1, x2, y2] bounding box."""
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

if __name__ == "__main__":
    # Integration logic verification
    tracker = AirportTracker()
    dummy_detections = [
        {"bbox": [50, 50, 150, 300], "class_id": 0, "confidence": 0.9}, # Person
        {"bbox": [160, 200, 210, 250], "class_id": 28, "confidence": 0.8} # Suitcase
    ]
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Note: DeepSORT update requires features, so results in a dummy frame may vary.
    results = tracker.update(dummy_detections, dummy_frame)
    print("Tracking results with associations:")
    print(results)
