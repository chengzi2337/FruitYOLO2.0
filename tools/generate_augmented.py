#!/usr/bin/env python3
"""Generate augmented images for training set using Albumentations and update YOLO labels.
Output augmented images/labels into Dataset_resplit_aug/ (keeps original files and adds aug_ suffix files).
"""
import os
from pathlib import Path
import argparse
import random
import shutil

def ensure_dirs(p):
    p.mkdir(parents=True, exist_ok=True)

def copy_originals(src_root, dst_root):
    # copy images and labels preserving structure
    for sub in ['images/train','labels/train']:
        s = Path(src_root)/sub
        d = Path(dst_root)/sub
        if not s.exists():
            continue
        ensure_dirs(d)
        for f in s.iterdir():
            if f.is_file():
                shutil.copy2(f, d/f.name)

def parse_yolo_label(path):
    # returns list of (class, x_center, y_center, w, h) floats
    res = []
    try:
        with open(path,'r',encoding='utf-8') as f:
            for l in f:
                toks = l.strip().split()
                if not toks: continue
                cls = int(float(toks[0]))
                coords = list(map(float, toks[1:5]))
                res.append((cls, *coords))
    except Exception:
        pass
    return res

def write_yolo_label(path, items):
    with open(path,'w',encoding='utf-8') as f:
        for it in items:
            cls = int(it[0])
            coords = [f'{x:.6f}' for x in it[1:]]
            f.write(str(cls) + ' ' + ' '.join(coords) + '\n')

def transform_bbox_yolo(bbox, aug, img_w, img_h):
    # bbox: (cls, x,y,w,h) normalized
    # convert to albumentations format (x_min,y_min,x_max,y_max absolute)
    cls, x, y, w, h = bbox
    x_c = x*img_w
    y_c = y*img_h
    bw = w*img_w
    bh = h*img_h
    x_min = x_c - bw/2
    y_min = y_c - bh/2
    x_max = x_c + bw/2
    y_max = y_c + bh/2
    # apply augmentation using albumentations (we'll do via callback)
    return (cls, x_min, y_min, x_max, y_max)

def bbox_to_yolo(coords, img_w, img_h):
    x_min,y_min,x_max,y_max = coords
    bw = x_max - x_min
    bh = y_max - y_min
    x_c = x_min + bw/2
    y_c = y_min + bh/2
    return (x_c/img_w, y_c/img_h, bw/img_w, bh/img_h)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='Dataset_resplit', help='source resplit dataset root')
    parser.add_argument('--out', default='Dataset_resplit_aug', help='output augmented dataset root')
    parser.add_argument('--factor', type=int, default=1, help='augmentation factor per image')
    args = parser.parse_args()

    try:
        import cv2
        import albumentations as A
    except Exception as e:
        print('Required packages missing: opencv-python, albumentations')
        raise

    src = Path(args.src)
    out = Path(args.out)
    # copy originals
    for d in ['images/train','images/val','images/test','labels/train','labels/val','labels/test']:
        srcd = src/d
        outd = out/d
        if srcd.exists():
            outd.mkdir(parents=True, exist_ok=True)
            if 'images' in d or 'labels' in d:
                # copy originals for non-train as well
                for f in srcd.iterdir():
                    if f.is_file():
                        shutil.copy2(f, outd/f.name)

    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.GaussNoise(p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    img_dir = src/'images'/'train'
    lab_dir = src/'labels'/'train'
    out_img_dir = out/'images'/'train'
    out_lab_dir = out/'labels'/'train'

    for img_path in img_dir.iterdir():
        if not img_path.is_file(): continue
        lab_path = lab_dir/(img_path.stem + '.txt')
        # read image
        img = cv2.imread(str(img_path))
        if img is None: continue
        h,w = img.shape[:2]
        labels = parse_yolo_label(lab_path) if lab_path.exists() else []
        # convert labels to pascal_voc absolute boxes
        bboxes = []
        cat_ids = []
        for it in labels:
            cls = it[0]
            x,y,ww,hh = it[1],it[2],it[3],it[4]
            x_c = x*w
            y_c = y*h
            bw = ww*w
            bh = hh*h
            x_min = max(0, x_c - bw/2)
            y_min = max(0, y_c - bh/2)
            x_max = min(w, x_c + bw/2)
            y_max = min(h, y_c + bh/2)
            bboxes.append([x_min,y_min,x_max,y_max])
            cat_ids.append(int(cls))

        # keep original already copied; produce augmented copies
        for i in range(args.factor):
            out_name = f"{img_path.stem}_aug{i}.jpg"
            if bboxes:
                try:
                    augmented = aug(image=img, bboxes=bboxes, category_ids=cat_ids)
                    aug_img = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_cat = augmented['category_ids']
                except Exception:
                    aug_img = img
                    aug_bboxes = bboxes
                    aug_cat = cat_ids
            else:
                # no bboxes
                augmented = aug(image=img)
                aug_img = augmented['image']
                aug_bboxes = []
                aug_cat = []

            out_fp = out_img_dir/out_name
            cv2.imwrite(str(out_fp), aug_img)
            # write label
            out_label = out_lab_dir/(out_fp.stem + '.txt')
            with open(out_label,'w',encoding='utf-8') as fh:
                for cid, box in zip(aug_cat, aug_bboxes):
                    x_min,y_min,x_max,y_max = box
                    # convert to yolo normalized
                    xc = (x_min + x_max)/2.0 / w
                    yc = (y_min + y_max)/2.0 / h
                    bw = (x_max - x_min)/w
                    bh = (y_max - y_min)/h
                    fh.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    # copy classes.txt
    if Path('classes.txt').exists():
        shutil.copy2('classes.txt', out/'classes.txt')

    # write data.yaml
    names = []
    if Path('classes.txt').exists():
        with open('classes.txt','r',encoding='utf-8') as f:
            names = [l.strip() for l in f if l.strip()]
    data_yaml = out/'data.yaml'
    with open(data_yaml,'w',encoding='utf-8') as f:
        f.write(f"train: {str(Path('images/train').as_posix())}\n")
        f.write(f"val: {str(Path('images/val').as_posix())}\n")
        f.write(f"test: {str(Path('images/test').as_posix())}\n\n")
        f.write(f"nc: {len(names)}\n")
        f.write('names: ' + str(names) + '\n')

    print('Augmentation complete ->', out)

if __name__ == '__main__':
    main()
