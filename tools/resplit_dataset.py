#!/usr/bin/env python3
"""Resplit dataset into stratified train/val/test (per-class) and write YOLO data.yaml for new dataset.
Creates directory `Dataset_resplit/` with images/labels subfolders.
Default split: 80% train, 10% val, 10% test (per-class stratified).
"""
import os
import shutil
import random
from pathlib import Path
import argparse

def read_label_classes(label_path):
    # read first token (class) from label file; return set of class ids in file
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            classes = set()
            for l in f:
                toks = l.strip().split()
                if not toks: continue
                classes.add(int(float(toks[0])))
            return classes
    except Exception:
        return set()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_images', default='Dataset_Original/images', help='source images root with train/val subdirs')
    parser.add_argument('--src_labels', default='Dataset_Original/labels', help='source labels root with train/val subdirs')
    parser.add_argument('--out', default='Dataset_resplit', help='output root')
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in ['images/train','images/val','images/test','labels/train','labels/val','labels/test']:
        (out / d).mkdir(parents=True, exist_ok=True)

    # collect all image-label pairs from src train+val
    src_imgs = []
    for split in ['train','val']:
        img_dir = Path(args.src_images) / split
        lab_dir = Path(args.src_labels) / split
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in ['.jpg','.jpeg','.png','bmp']:
                continue
            lab = lab_dir / (img.stem + '.txt')
            src_imgs.append((str(img), str(lab) if lab.exists() else None))

    # map class -> list of image entries (use first class in file or multiple entries duplicated)
    class_to_items = {}
    for img, lab in src_imgs:
        if lab and Path(lab).exists():
            classes = read_label_classes(lab)
            if not classes:
                # put into class -1 bucket
                class_to_items.setdefault(-1, []).append((img, lab))
            else:
                for c in classes:
                    class_to_items.setdefault(c, []).append((img, lab))
        else:
            class_to_items.setdefault(-1, []).append((img, lab))

    # For images appearing in multiple class buckets, we risk duplicates; instead create a unique list and choose main class as first token
    unique_items = {}
    for img, lab in src_imgs:
        if img in unique_items:
            continue
        if lab and Path(lab).exists():
            classes = read_label_classes(lab)
            main = min(classes) if classes else -1
        else:
            main = -1
        unique_items[img] = (img, lab, main)

    # rebuild class mapping
    class_to_items = {}
    for img,lab,main in unique_items.values():
        class_to_items.setdefault(main, []).append((img,lab))

    # split per class
    train_list=[]
    val_list=[]
    test_list=[]
    for cls, items in class_to_items.items():
        random.shuffle(items)
        n = len(items)
        ntrain = int(n * args.train_frac)
        nval = int(n * args.val_frac)
        # ensure at least one in each if possible
        if n > 0 and ntrain == 0:
            ntrain = max(1, n-2)
        if n - ntrain - nval <= 0 and n - ntrain > 0:
            nval = max(1, n - ntrain - 1)
        train_items = items[:ntrain]
        val_items = items[ntrain:ntrain+nval]
        test_items = items[ntrain+nval:]
        train_list.extend(train_items)
        val_list.extend(val_items)
        test_list.extend(test_items)

    # deduplicate if same image assigned multiple times
    def copy_list(lst, target_img_dir, target_lab_dir):
        seen=set()
        for img, lab in lst:
            if img in seen: continue
            seen.add(img)
            src_img = Path(img)
            dst_img = Path(args.out)/target_img_dir/src_img.name
            shutil.copy2(src_img, dst_img)
            if lab and Path(lab).exists():
                dst_lab = Path(args.out)/target_lab_dir/(Path(lab).stem + '.txt')
                shutil.copy2(lab, dst_lab)

    copy_list(train_list, 'images/train', 'labels/train')
    copy_list(val_list, 'images/val', 'labels/val')
    copy_list(test_list, 'images/test', 'labels/test')

    # write data.yaml
    classes_file = Path('classes.txt')
    names = []
    if classes_file.exists():
        with classes_file.open('r', encoding='utf-8') as f:
            names = [l.strip() for l in f.readlines() if l.strip()]

    data_yaml = Path(args.out)/'data.yaml'
    with data_yaml.open('w', encoding='utf-8') as f:
        f.write(f"train: {str(Path('images/train').as_posix())}\n")
        f.write(f"val: {str(Path('images/val').as_posix())}\n")
        f.write(f"test: {str(Path('images/test').as_posix())}\n\n")
        f.write(f"nc: {len(names)}\n")
        f.write('names: ' + str(names) + '\n')

    print('Wrote resplit dataset to', args.out)

if __name__ == '__main__':
    main()
