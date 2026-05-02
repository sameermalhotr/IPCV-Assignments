import cv2
import numpy as np
import os

def setup():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    print("TRAFFIC MONITORING SYSTEM INITIALIZED")

def edge_detection(gray):
    # Sobel Operator
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_final = np.uint8(np.absolute(sobel_combined))
    
    # Canny Edge Detector
    canny = cv2.Canny(gray, 100, 200)
    
    cv2.imwrite('outputs/edge_sobel.png', sobel_final)
    cv2.imwrite('outputs/edge_canny.png', canny)
    return canny

def object_representation(gray, canny_edges):
    contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print(f"\n--- Object Representation Metrics ---")
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter noise to find vehicles
            count += 1
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"Object {count}: Area={area:.2f}, Perimeter={perimeter:.2f}")
            
    cv2.imwrite('outputs/objects_detected.png', output_img)

def feature_extraction(gray):
    # Using ORB (Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    feature_img = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0))
    cv2.imwrite('outputs/features_orb.png', feature_img)
    print(f"\n--- Feature Extraction ---")
    print(f"Total Keypoints Detected: {len(keypoints)}")

def process_traffic_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: {img_path} not found.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = edge_detection(gray)
    object_representation(gray, canny)
    feature_extraction(gray)
    print("\nProcessing complete. Check 'outputs/' folder.")

if __name__ == "__main__":
    setup()
    # Replace 'traffic.jpg' with your traffic image file path
    process_traffic_image('traffic.jpg')