import cv2
from TeaDetector import YOLOProcessor # Imports the class from TeaDetector.py

def main():

    # Define global configuration constants
    model_path = "Model/Detector_YOLO26m_v3.0.pt"
    image_path = "Data/Tea_Dataset/Tea_Making/dr_1_857.jpg"
    
    # 1. Initialize
    detector = YOLOProcessor(model_path)
    
    # 2. Extract 
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    crops, boxes = detector.extract_leaf_crops(image_path)
    
    # 3. Visualize
    print(f"Found {len(boxes)} leaves.")
    detector.visualize_detections(img_rgb, boxes)

if __name__ == "__main__":
    main()