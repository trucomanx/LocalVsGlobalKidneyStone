import os
import argparse
import json

from dp_utils.image_processing import (
    np_zero_image,
    image_to_list_patch,
    save_patches_list,
    generate_images_patches,
)

from dp_utils.csv_processing import (
    datasets_gen,
    datasets_gen_kfold
)

def generate_dataset(ds_path,base_out_path,patch_size,classes=["stone"]):

    out_path = os.path.join(base_out_path,f'dataset-{patch_size}')
    
    generate_images_patches(class_name_list=classes, 
                            img_patch_size=patch_size, 
                            ds_path=ds_path,
                            out_path=out_path )
    
    if os.path.exists(out_path):
        X, y = datasets_gen(out_path=out_path)   
        
        report = {"with-stone": len(y)-y.count(0) ,"without-stone": y.count(0)}
        with open(os.path.join(out_path, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
    
        datasets_gen_kfold(X, y, num_folds=5, out_path=out_path)

if __name__ == "__main__":
    
    ds_path = "data/dataset-source/"
    out_path = 'data/dataset-article/'
    CLASSES = ["stone"]

    parser = argparse.ArgumentParser(description="Image patches processing")
    parser.add_argument("-ps", "--patch_size", required=True, type=int)
    args = parser.parse_args()
    
    patch_size = args.patch_size
    
    generate_dataset(ds_path,out_path,patch_size,CLASSES)

