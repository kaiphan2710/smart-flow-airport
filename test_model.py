import cv2
import os
from src.vision.detector import AirportDetector

def test_yolo_on_image(image_path, output_path="output_test.jpg"):
    print(f"Đang load mô hình và test trên ảnh: {image_path}")
    
    # 1. Khởi tạo mô hình
    detector = AirportDetector(model_path='yolov8n.pt')
    
    # 2. Đọc ảnh
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {image_path}")
        return
        
    frame = cv2.imread(image_path)
    
    # 3. Chạy model dự đoán
    detections = detector.process_frame(frame)
    print(f"Phát hiện được {len(detections)} đối tượng.")
    
    # 4. Vẽ khung hình (Bounding Box) lên ảnh để kiểm tra bằng mắt thường
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = det['confidence']
        cls_name = det['class_name']
        
        # Chọn màu: Người (Xanh lá), Hành lý (Đỏ/Cam)
        color = (0, 255, 0) if cls_name == "Person" else (0, 165, 255)
        
        # Vẽ hình chữ nhật và in tên
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 5. Lưu ảnh kết quả
    cv2.imwrite(output_path, frame)
    print(f"Đã lưu ảnh kết quả tại: {output_path}. Hãy mở ra xem nhé!")

if __name__ == "__main__":
    # Lấy thử một ảnh có người mang hành lý trong dataset của bạn
    test_image = "awesome-reid-dataset/imgs/eg_market.jpg" 
    
    test_yolo_on_image(test_image)