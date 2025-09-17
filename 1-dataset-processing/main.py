import os
import argparse
import json

from utils.image_processing import (
    np_zero_image,
    image_to_list_patch,
    save_patches_list,
    generate_images_patches,
)

from utils.csv_processing import (
    datasets_gen,
    datasets_gen_kfold
)

ds_path = "data/dataset-source/"
out_path = 'data/dataset-article/'

PATCH_SIZE = 224
CLASSES = ["stone"]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Image patches processing")
    parser.add_argument("-ps", "--patch_size", required=True, type=int)
    args = parser.parse_args()
    
    patch_size = args.patch_size
    
    generate_images_patches(class_name_list=CLASSES, 
                            img_pacth_size=patch_size, 
                            ds_path=ds_path )
    
    out_path = f"{out_path}dataset-{patch_size}"
    
    if os.path.exists(out_path):
        X, y = datasets_gen(out_path=out_path)   
        
        report = {"with-stone": len(y)-y.count(0) ,"without-stone": y.count(0)}
        with open(os.path.join(out_path, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
    
        datasets_gen_kfold(X, y, num_folds=5, out_path=out_path)


