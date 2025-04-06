import os
import requests
import tarfile
import shutil
from tqdm import tqdm

def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def prepare_imagenet_lt(imagenet_train_dir, imagenet_val_dir, output_dir):
    """Main function to prepare ImageNet-LT dataset"""
    
    # Create output directories
    lt_train_dir = os.path.join(output_dir, "ImageNet-LT", "train")
    lt_val_dir = os.path.join(output_dir, "ImageNet-LT", "val")
    os.makedirs(lt_train_dir, exist_ok=True)
    os.makedirs(lt_val_dir, exist_ok=True)

    # Download split files
    base_url = "https://raw.githubusercontent.com/zhmiao/OpenLongTailRecognition-OLTR/master/data/metadata/"
    split_files = {
        "train": "ImageNet_LT_train.txt",
        "val": "ImageNet_LT_test.txt"
    }

    # Download and process splits
    for split, filename in split_files.items():
        print(f"\nProcessing {split} split...")
        url = base_url + filename
        local_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            download_file(url, local_path)

        # Create class directories
        with open(local_path) as f:
            lines = f.readlines()
            
        # Create category buckets
        category_counts = {}
        for line in tqdm(lines, desc="Organizing classes"):
            path, class_idx = line.strip().split()
            class_idx = int(class_idx)
            category_counts[class_idx] = category_counts.get(class_idx, 0) + 1

        # Create symlinks for LT dataset
        for line in tqdm(lines, desc=f"Creating {split} links"):
            src_path, class_idx = line.strip().split()
            class_idx = int(class_idx)
            
            # Get original image path
            if split == "train":
                original_path = os.path.join(imagenet_train_dir, src_path)
            else:
                original_path = os.path.join(imagenet_val_dir, src_path)
                
            # Create category directory
            category = "many" if category_counts[class_idx] > 100 else \
                      "medium" if 20 <= category_counts[class_idx] <= 100 else "rare"
                      
            dest_dir = os.path.join(output_dir, "ImageNet-LT", split, category, str(class_idx))
            os.makedirs(dest_dir, exist_ok=True)
            
            # Create symlink
            dest_path = os.path.join(dest_dir, os.path.basename(src_path))
            if not os.path.exists(dest_path):
                os.symlink(original_path, dest_path)

    print("\nDataset preparation complete!")
    print(f"ImageNet-LT directory structure created at: {os.path.join(output_dir, 'ImageNet-LT')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare ImageNet-LT dataset')
    parser.add_argument('--imagenet_train', type=str, required=True,
                       help='Path to original ImageNet training set')
    parser.add_argument('--imagenet_val', type=str, required=True,
                       help='Path to original ImageNet validation set')
    parser.add_argument('--output_dir', type=str, default="./data",
                       help='Output directory for processed dataset')
    
    args = parser.parse_args()

    # Verify original ImageNet paths
    if not os.path.exists(args.imagenet_train):
        raise FileNotFoundError(f"ImageNet training directory not found at {args.imagenet_train}")
    if not os.path.exists(args.imagenet_val):
        raise FileNotFoundError(f"ImageNet validation directory not found at {args.imagenet_val}")

    prepare_imagenet_lt(
        imagenet_train_dir=args.imagenet_train,
        imagenet_val_dir=args.imagenet_val,
        output_dir=args.output_dir
    )
