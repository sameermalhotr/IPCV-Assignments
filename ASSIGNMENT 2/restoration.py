import cv2
import numpy as np
import os
import math

def setup():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    print("SURVEILLANCE IMAGE RESTORATION SYSTEM READY")

def calculate_metrics(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return 0, 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return mse, psnr

def add_gaussian_noise(image):
    row, col = image.shape
    mean = 0
    var = 0.01
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col)) * 255
    noisy = np.clip(image + gauss.reshape(row, col), 0, 255)
    return noisy.astype(np.uint8)

def add_salt_and_pepper(image):
    row, col = image.shape
    s_vs_p = 0.5
    amount = 0.04
    out = np.copy(image)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out

def run_restoration(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: {img_path} not found")
        return

    cv2.imwrite('outputs/original.png', img)

    g_noisy = add_gaussian_noise(img)
    sp_noisy = add_salt_and_pepper(img)
    cv2.imwrite('outputs/gaussian_noisy.png', g_noisy)
    cv2.imwrite('outputs/sp_noisy.png', sp_noisy)

    filters = {
        "Mean": lambda x: cv2.blur(x, (5, 5)),
        "Median": lambda x: cv2.medianBlur(x, 5),
        "Gaussian": lambda x: cv2.GaussianBlur(x, (5, 5), 0)
    }

    results = []
    for noise_name, noisy_img in [("Gaussian", g_noisy), ("Salt-and-Pepper", sp_noisy)]:
        for f_name, f_func in filters.items():
            restored = f_func(noisy_img)
            mse, psnr = calculate_metrics(img, restored)
            results.append((noise_name, f_name, mse, psnr))
            cv2.imwrite(f'outputs/restored_{noise_name}_{f_name}.png', restored)

    print(f"\n{'Noise Type':<20} | {'Filter':<10} | {'MSE':<10} | {'PSNR (dB)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r[0]:<20} | {r[1]:<10} | {r[2]:<10.2f} | {r[3]:<10.2f}")

if __name__ == "__main__":
    setup()
    # Replace 'surveillance.jpg' with your image file
    run_restoration('surveillance.jpg')