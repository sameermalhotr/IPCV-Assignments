import cv2
import numpy as np
import os

def rle_encode(img):
    flat = img.flatten()
    n = len(flat)
    if n == 0: return []
    encoding = []
    prev = flat[0]
    count = 1
    for i in range(1, n):
        if flat[i] == prev:
            count += 1
        else:
            encoding.append((prev, count))
            prev = flat[i]
            count = 1
    encoding.append((prev, count))
    return encoding

def setup():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

def process_medical_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: {path} not found")
        return
    
    cv2.imwrite('outputs/original.png', img)
    
    encoded = rle_encode(img)
    orig_size = img.size
    comp_size = len(encoded) * 2 
    ratio = orig_size / comp_size
    savings = (1 - (comp_size / orig_size)) * 100
    
    print(f"--- Compression Results ---")
    print(f"Original Pixels: {orig_size}")
    print(f"RLE Pairs: {len(encoded)}")
    print(f"Compression Ratio: {ratio:.2f}:1")
    print(f"Storage Savings: {savings:.2f}%")
    
    ret, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret_otsu, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cv2.imwrite('outputs/global_thresh.png', global_thresh)
    cv2.imwrite('outputs/otsu_thresh.png', otsu_thresh)
    
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(otsu_thresh, kernel, iterations=1)
    dilation = cv2.dilate(otsu_thresh, kernel, iterations=1)
    refined = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
    
    cv2.imwrite('outputs/refined_segmentation.png', refined)
    print("\nProcessing complete. Images saved in 'outputs/' folder.")

if __name__ == "__main__":
    setup()
    # Replace 'medical.png' with your actual image file path
    process_medical_image('medical.png')