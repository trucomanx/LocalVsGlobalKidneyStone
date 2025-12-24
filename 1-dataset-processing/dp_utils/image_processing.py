import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from natsort import natsorted

try:
    # Jupyter / Colab
    from tqdm.notebook import tqdm
except ImportError:
    # Terminal / script normal
    from tqdm import tqdm


def np_zero_image(ds_path: str) -> None:
    
    for i, image in enumerate(glob.glob(os.path.join(ds_path,"normal","image","*jpg"))):
      # read image in numpy array format
      img_normal = cv2.imread(image)
      
      # zero all pixels values image shape is (w, h, c)
      # convert image to black
      img_zeros = np.zeros(img_normal.shape)
      
      # define the path to write the image
      out_path = os.path.join(ds_path,"normal","label", f"Normal- ({i+1}).jpg")
      cv2.imwrite(out_path,  img_zeros)
      
      plt.imshow(img_normal)


def image_to_list_patch_old(image, label_image, patch_size):
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

import numpy as np

def image_to_list_patch(image, label_image, patch_size, rest=0.25):
    """
    Divide uma imagem e seu respectivo label em patches sobrepostos.
    Permite lidar com sobras menores que o tamanho do patch.

    Args:
        image (ndarray): imagem original (H, W, C) ou (H, W).
        label_image (ndarray): máscara/label correspondente (H, W) ou (H, W, C_label).
        patch_size (int): tamanho do patch quadrado.
        rest (float): proporção do patch_size que decide se a sobra é ignorada (0 a 1).

    Returns:
        (list, list): listas com patches da imagem e patches do label.
    """
    my_image_list = []
    my_label_list = []

    step = patch_size // 2
    h, w = image.shape[:2]

    # Lista de coordenadas iniciais
    y0_list = list(range(0, h - patch_size + 1, step))
    x0_list = list(range(0, w - patch_size + 1, step))

    # Checa sobra em Y
    if h - (y0_list[-1] + patch_size) > rest * patch_size:
        y0_list.append(h - patch_size)
    # Checa sobra em X
    if w - (x0_list[-1] + patch_size) > rest * patch_size:
        x0_list.append(w - patch_size)

    patch_num = 0
    for y0 in y0_list:
        for x0 in x0_list:
            y1 = y0 + patch_size
            x1 = x0 + patch_size

            # Patch vazio, preservando dimensões da imagem
            if image.ndim == 3:
                patch_image = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
            else:
                patch_image = np.zeros((patch_size, patch_size), dtype=image.dtype)

            if label_image.ndim == 3:
                patch_label = np.zeros((patch_size, patch_size, label_image.shape[2]), dtype=label_image.dtype)
            else:
                patch_label = np.zeros((patch_size, patch_size), dtype=label_image.dtype)

            # Limites reais do recorte (não ultrapassa borda)
            y1_real = min(y1, h)
            x1_real = min(x1, w)

            # Copia a parte real da imagem/máscara
            patch_image[0:y1_real - y0, 0:x1_real - x0, ...] = image[y0:y1_real, x0:x1_real, ...]
            patch_label[0:y1_real - y0, 0:x1_real - x0, ...] = label_image[y0:y1_real, x0:x1_real, ...]

            patch_num += 1
            my_image_list.append(patch_image)
            my_label_list.append(patch_label)

    return my_image_list, my_label_list


def save_patches_list(  img_list: list, 
                        img_label_list: list, 
                        patch_size: int, 
                        image_index: str, 
                        class_name: str,
                        out_path:str) -> None:
  
  # define minimum threshold value for max pixels in label images (0 - 255)
  min_threshold = 10 
  
  #print(f"Saving image with id:{image_index} from patch {patch_size}.")
  
  # [numpy.array,numpy.array,numpy.array, ....]  [0,1,1,0,0,0,0,1]
  # save in dir

  os.makedirs(os.path.join(out_path,"with-stone"), exist_ok=True)
  os.makedirs(os.path.join(out_path,"without-stone"), exist_ok=True)

  for i, (label_image, image) in enumerate(zip(img_label_list, img_list)):
    # print(f"Patch {i}: label={np.max(label_image)}, image={image.shape}")
    # print('labels: ', np.max(label_image))
    filename=f"image{class_name}-{image_index}-patch{i}.png"
    if np.max(label_image) > min_threshold:
      fpath=os.path.join(out_path,"with-stone",filename)
    else:
      fpath=os.path.join(out_path,"without-stone",filename)
    
    cv2.imwrite(fpath, image)
 

def generate_images_patches(class_name_list: list[str], 
                            img_patch_size: int, 
                            ds_path: str,
                            out_path: str) -> None:
    
    print("")
    print("Working patch_size:", img_patch_size)
    
    for class_name in class_name_list:
        
        image_type = '.tif' if class_name == 'stone' else '.jpg'
        label_type = '.tif' if class_name == 'stone' else '.jpg'
    
        # get all image paths
        image_dir = os.path.join(ds_path, class_name, "image")
        label_dir = os.path.join(ds_path, class_name, "label")
        
        print("Generating class:", class_name)
        
        # Lista e ordena naturalmente
        fpath=os.path.join(image_dir,f"*{image_type}")
        image_paths = natsorted(glob.glob(fpath))

        # Itera com tqdm
        for image_index, image_path in enumerate(tqdm(image_paths, desc="Processando imagens")):
        
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
        
            img_list, img_label_list =  image_to_list_patch(img, img_label, patch_size=img_patch_size)
            
            save_patches_list(
                img_list, 
                img_label_list=img_label_list, 
                patch_size=img_patch_size, 
                image_index=filename, #image_index, 
                class_name=class_name,
                out_path = out_path
            )
        

