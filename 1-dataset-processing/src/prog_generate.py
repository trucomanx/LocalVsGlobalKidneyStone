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
    os.makedirs(out_path,exist_ok=True)
    
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

    return out_path


if __name__ == "__main__":

    CLASSES = ["stone"]

    parser = argparse.ArgumentParser(description="Image patches processing")

    parser.add_argument(
        "-ds", "--dataset_source",
        required=True,
        type=str,
        help="Source dataset directory (example: data/dataset-source)"
    )

    parser.add_argument(
        "-do", "--dataset_output",
        required=True,
        type=str,
        help="Output dataset directory (example: data/dataset-article)"
    )

    parser.add_argument(
        "-ps", "--patch_size",
        required=True,
        type=int,
        help="Image patch size"
    )

    args = parser.parse_args()

    generate_dataset(
        ds_path=args.dataset_source,
        base_out_path=args.dataset_output,
        patch_size=args.patch_size,
        classes=CLASSES
    )

