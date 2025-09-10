import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def np_zero_image(ds_path: str) -> None:
    for i, image in enumerate(glob.glob(f"{ds_path}/normal/image/*jpg")):
      # read image in numpy array format
      img_normal = cv2.imread(image)
      
      # zero all pixels values image shape is (w, h, c)
      # convert image to black
      img_zeros = np.zeros(img_normal.shape)
      
      # define the path to write the image
      out_path = os.path.join(f"{ds_path}/normal/label/", f"Normal- ({i+1}).jpg")
      cv2.imwrite(out_path,  img_zeros)
      
      plt.imshow(img_normal)


def image_to_list_patch(image, label_image, patch_size):
    """
    Divide uma imagem e seu respectivo label em patches sobrepostos.

    Args:
        image (ndarray): imagem original (H, W, C) ou (H, W).
        label (ndarray): máscara/label correspondente (H, W).
        patch_size (int): tamanho do patch quadrado.

    Returns:
        (list, list): listas com patches da imagem e patches do label.
    """
    my_image_list = []
    my_label_list = []

    patch_num = 0
    for y0 in range(0, image.shape[0] - patch_size + 1, patch_size // 2):
        for x0 in range(0, image.shape[1] - patch_size + 1, patch_size // 2):
            x1 = x0 + patch_size
            y1 = y0 + patch_size

            patch_image = image[y0:y1, x0:x1]
            patch_label = label_image[y0:y1, x0:x1]

            patch_num += 1
            # print(f"Patch {patch_num}: shape={patch_image.shape}, x=({x0},{x1}), y=({y0},{y1})")

            my_image_list.append(patch_image)
            my_label_list.append(patch_label)

    return my_image_list, my_label_list


def save_patches_list(img_list: list, img_label_list: list, patch_size: int, image_index: int, class_name: str) -> None:
  
  # define minimum threshold value for max pixels in label images (0 - 255)
  min_threshold = 10 
  
  print("\nSaving image to output dir...")
  
  # [numpy.array,numpy.array,numpy.array, ....]  [0,1,1,0,0,0,0,1]
  # save in dir

  out_path = f'data/dataset-article/dataset-{patch_size}/'

  os.makedirs(f"{out_path}/with-stone", exist_ok=True)
  # os.makedirs(f"{out_path}/with-stone/label", exist_ok=True)
  # os.makedirs(f"{out_path}/without-stone/image", exist_ok=True)
  os.makedirs(f"{out_path}/without-stone", exist_ok=True)

  for i, (label_image, image) in enumerate(zip(img_label_list, img_list)):

    # print(f"Patch {i}: label={np.max(label_image)}, image={image.shape}")

    # print('labels: ', np.max(label_image))
    
    if np.max(label_image) > min_threshold:
      cv2.imwrite(f"{out_path}/with-stone/image{class_name}{image_index}-patch{i}.png", image)
      # cv2.imwrite(f"{out_path}/with-stone/label{class_name}{image_index}-patch{i}.png", label_image)
    else:
      cv2.imwrite(f"{out_path}/without-stone/image{class_name}{image_index}-patch{i}.png", image)
      # cv2.imwrite(f"{out_path}/without-stone/label{class_name}{image_index}-patch{i}.png", label_image)
      

def generate_images_patches(class_name_list: list[str], img_pacth_size: int, ds_path: str) -> None:
    
    for class_name in class_name_list:
        
        image_type = '.tif' if class_name == 'stone' else '.jpg'
        label_type = '.tif' if class_name == 'stone' else '.jpg'
    
        # get all image paths
        image_dir = os.path.join(ds_path, class_name, "image")
        label_dir = os.path.join(ds_path, class_name, "label")
        
        for image_index, image_path in enumerate(glob.glob(f"{image_dir}/*{image_type}")):
        
            # get filename only (without extension)
            filename = os.path.splitext(os.path.basename(image_path))[0]

            # corresponding label path (assuming same filename but maybe PNG or JPG)
            label_path = os.path.join(label_dir, filename + label_type)

            if not os.path.exists(label_path):
                print(f"Label not found for {filename}")
                continue
            
            # read image in numpy array format
            img = cv2.imread(image_path)
            img_label = cv2.imread(label_path)
        
            img_list, img_label_list =  image_to_list_patch(img, img_label, patch_size=img_pacth_size)
            
            save_patches_list(
                img_list, 
                img_label_list=img_label_list, 
                patch_size=img_pacth_size, 
                image_index=image_index, 
                class_name=class_name
            )
        