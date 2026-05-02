import cv2
import numpy as np
import os
import math

def setup():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

def calculate_metrics(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 0, 100
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return mse, psnr

def enhance_image(gray):
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced

def restore_image(img):
    return cv2.medianBlur(img, 5)

def segment_image(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    refined = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return refined

def extract_features(img):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    out = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
    return out, len(kp)

def run_pipeline(img_path):
    orig = cv2.imread(img_path)
    if orig is None:
        print("File not found.")
        return
    
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_image(gray)
    restored = restore_image(enhanced)
    segmented = segment_image(restored)
    featured, kp_count = extract_features(restored)
    
    mse, psnr = calculate_metrics(gray, restored)
    
    cv2.imwrite('outputs/1_gray.png', gray)
    cv2.imwrite('outputs/2_enhanced.png', enhanced)
    cv2.imwrite('outputs/3_restored.png', restored)
    cv2.imwrite('outputs/4_segmented.png', segmented)
    cv2.imwrite('outputs/5_features.png', featured)
    
    print("--- PIPELINE METRICS ---")
    print(f"MSE (Orig vs Restored): {mse:.2f}")
    print(f"PSNR (Orig vs Restored): {psnr:.2f} dB")
    print(f"Features Detected: {kp_count}")

if __name__ == "__main__":
    setup()
    run_pipeline('input.jpg')