import cv2
import numpy as np
import os
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, target_size=(240, 320)):
    """Process IR images with normalization, interpolation and colormap"""
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in tqdm(os.listdir(input_dir)):
        if img_file.endswith('.png'):
            img_path = os.path.join(input_dir, img_file)
            ir_image = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            
            img_norm = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)
            
            img_upscaled = cv2.resize(img_norm, target_size, 
                                    interpolation=cv2.INTER_CUBIC)
            
            img_color = cv2.applyColorMap(img_upscaled.astype(np.uint8), 
                                        cv2.COLORMAP_JET)
            
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, img_color)

if __name__ == '__main__':
    preprocess_images(
        input_dir= 'GIRASOL_DATASET/2017_12_18/infrared',
        output_dir='data/processed/2017_12_18'
    )