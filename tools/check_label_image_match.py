#!/usr/bin/env python3
"""Check strict correspondence between images and YOLO label files.
Outputs per-split counts, mismatches and sample problematic entries.
"""
import argparse
from pathlib import Path
import re


def check_split(root: Path, split: str):
    img_dir = root / 'images' / split
    lab_dir = root / 'labels' / split
    exts = ('.jpg', '.jpeg', '.png')
    img_stems = set()
    lab_stems = set()
    if img_dir.exists():
        for p in img_dir.iterdir():
            if p.suffix.lower() in exts and p.is_file():
                img_stems.add(p.stem)
    if lab_dir.exists():
        for p in lab_dir.iterdir():
            if p.suffix.lower() == '.txt' and p.is_file():
                lab_stems.add(p.stem)

    labs_without_img = sorted(list(lab_stems - img_stems))
    imgs_without_lab = sorted(list(img_stems - lab_stems))

    # validate label contents
    malformed = []
    nonint = []
    total_label_files = 0
    if lab_dir.exists():
        for p in lab_dir.iterdir():
            if p.suffix.lower() != '.txt' or not p.is_file():
                continue
            total_label_files += 1
            try:
                lines = p.read_text(encoding='utf-8').splitlines()
            except Exception:
                malformed.append(f"UNREADABLE: {p}")
                continue
            for i, line in enumerate(lines):
                s = line.strip()
                if not s:
                    continue
                parts = re.split(r"\s+", s)
                if len(parts) < 5:
                    malformed.append(f"{p}: line {i+1} malformed: {s}")
                    if len(malformed) > 50:
                        break
                else:
                    if not re.fullmatch(r"\d+", parts[0]):
                        nonint.append(f"{p}: line {i+1} class_token='{parts[0]}'")
                        if len(nonint) > 50:
                            break
            if len(malformed) > 50 and len(nonint) > 50:
                break

    summary = {
        'split': split,
        'images': len(img_stems),
        'labels': total_label_files,
        'labs_without_img': len(labs_without_img),
        'imgs_without_lab': len(imgs_without_lab),
        'malformed_label_lines': len(malformed),
        'nonint_class_tokens': len(nonint),
        'sample_labs_without_img': labs_without_img[:20],
        'sample_imgs_without_lab': imgs_without_lab[:20],
        'sample_malformed': malformed[:20],
        'sample_nonint': nonint[:20],
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='Dataset_resplit_aug', help='dataset root')
    args = p.parse_args()
    root = Path(args.root)

    for split in ('train', 'val', 'test'):
        res = check_split(root, split)
        print('----', res['split'], '----')
        print(f"images: {res['images']}  labels: {res['labels']}")
        print(f"labs_without_img: {res['labs_without_img']}  imgs_without_lab: {res['imgs_without_lab']}")
        print(f"malformed_label_lines: {res['malformed_label_lines']}  nonint_class_tokens: {res['nonint_class_tokens']}")
        if res['sample_labs_without_img']:
            print('Sample labs_without_img:')
            for s in res['sample_labs_without_img']:
                print(' ', s)
        if res['sample_imgs_without_lab']:
            print('Sample imgs_without_lab:')
            for s in res['sample_imgs_without_lab']:
                print(' ', s)
        if res['sample_malformed']:
            print('Sample malformed:')
            for s in res['sample_malformed']:
                print(' ', s)
        if res['sample_nonint']:
            print('Sample nonint:')
            for s in res['sample_nonint']:
                print(' ', s)
        print()


if __name__ == '__main__':
    main()
