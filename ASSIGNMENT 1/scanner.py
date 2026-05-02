import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def setup_environment():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    print("="*50)
    print("SMART DOCUMENT SCANNER SYSTEM")
    print("="*50)

def acquire_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img_resized = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return img_resized, gray

def analyze_sampling(gray_img):
    res_256 = cv2.resize(gray_img, (256, 256), interpolation=cv2.INTER_AREA)
    res_128 = cv2.resize(gray_img, (128, 128), interpolation=cv2.INTER_AREA)
    
    s_512 = gray_img
    s_256 = cv2.resize(res_256, (512, 512), interpolation=cv2.INTER_NEAREST)
    s_128 = cv2.resize(res_128, (512, 512), interpolation=cv2.INTER_NEAREST)
    return s_512, s_256, s_128

def analyze_quantization(gray_img):
    q_8bit = gray_img
    q_4bit = (gray_img // 16) * 16
    q_2bit = (gray_img // 64) * 64
    return q_8bit, q_4bit, q_2bit

def display_and_save(original, s_results, q_results):
    titles = ['Original', '512x512', '256x256', '128x128', '8-bit', '4-bit', '2-bit']
    images = [original, s_results[0], s_results[1], s_results[2], q_results[0], q_results[1], q_results[2]]
    
    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i], cmap='gray' if i > 0 else None)
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/comparison_figure.png')
    plt.show()

if __name__ == "__main__":
    setup_environment()
    # Change 'document.jpg' to your actual file name
    try:
        orig, gray = acquire_image('document.jpg')
        s_res = analyze_sampling(gray)
        q_res = analyze_quantization(gray)
        display_and_save(orig, s_res, q_res)
    except Exception as e:
        print(f"Error: {e}")