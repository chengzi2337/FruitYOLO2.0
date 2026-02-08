#!/usr/bin/env python3
"""Fix augmented YOLO labels:
- convert class indices like '0.0' -> '0'
- normalize label lines to: int x y w h (floats for bbox)
- move label files without matching images to diagnostics/orphan_labels
"""
import argparse
from pathlib import Path
import shutil
import sys


def find_image_for_label(img_dir: Path, stem: str):
    # try exact stem with common extensions
    for ext in ('.jpg', '.jpeg', '.png'):
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    # try variants: append _aug or look for files starting with stem
    for ext in ('.jpg', '.jpeg', '.png'):
        p = img_dir / (stem + '_aug' + ext)
        if p.exists():
            return p
    # fuzzy: any file in img_dir that startswith stem
    for f in img_dir.glob(stem + '*'):
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            return f
    return None


def fix_labels(root: Path, orphan_dir: Path, classes_file: Path):
    imgs_root = root / 'images'
    labels_root = root / 'labels'
    orphan_dir.mkdir(parents=True, exist_ok=True)

    classes = []
    if classes_file.exists():
        classes = [l.strip() for l in classes_file.read_text(encoding='utf-8').splitlines() if l.strip()]
    nc = len(classes)

    total_files = 0
    fixed_lines = 0
    moved_orphans = 0
    malformed = 0
    orphan_list = []

    for lab_path in labels_root.rglob('*.txt'):
        total_files += 1
        rel = lab_path.relative_to(labels_root.parent)
        # determine corresponding image dir based on labels subdir
        # labels structure: labels/<split>/<file>.txt
        try:
            split = lab_path.parent.name
            img_dir = imgs_root / split
        except Exception:
            img_dir = imgs_root

        lines = [L.rstrip('\n') for L in lab_path.read_text(encoding='utf-8').splitlines()]
        new_lines = []
        bad = False
        for i, line in enumerate(lines):
            if not line or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                malformed += 1
                bad = True
                continue
            cls_token = parts[0]
            try:
                # allow floats like 0.0
                if '.' in cls_token:
                    cls_int = int(float(cls_token))
                else:
                    cls_int = int(cls_token)
            except Exception:
                malformed += 1
                bad = True
                continue
            # check range if classes loaded
            if nc > 0 and (cls_int < 0 or cls_int >= nc):
                # keep but flag as malformed
                malformed += 1
                bad = True
            # rest numbers to float
            try:
                rest = [f"{float(x):.6f}" for x in parts[1:5]]
            except Exception:
                malformed += 1
                bad = True
                continue
            new_lines.append(' '.join([str(cls_int)] + rest))
            if cls_token != str(cls_int):
                fixed_lines += 1

        # write fixed labels if any
        if new_lines and not bad:
            lab_path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
        elif new_lines and bad:
            # still write corrected lines to reduce downstream errors
            lab_path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')

        # check matching image
        stem = lab_path.stem
        img = find_image_for_label(img_dir, stem)
        if img is None:
            # try checking all image folders in imgs_root (sometimes labels in test refer to images in other folder)
            found = None
            for d in imgs_root.iterdir():
                if not d.is_dir():
                    continue
                f = find_image_for_label(d, stem)
                if f is not None:
                    found = f
                    break
            if found is None:
                # move orphan label
                target = orphan_dir / lab_path.name
                shutil.move(str(lab_path), str(target))
                moved_orphans += 1
                orphan_list.append(str(lab_path))

    # summary
    summary = [
        f"classes_count: {nc}",
        f"total_label_files: {total_files}",
        f"fixed_label_lines(class token changes): {fixed_lines}",
        f"malformed_label_lines: {malformed}",
        f"moved_orphan_labels: {moved_orphans}",
    ]
    return summary, orphan_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='Dataset_resplit_aug', help='dataset root')
    p.add_argument('--orphan', default='runs/detect/diagnostics/orphan_labels', help='where to move orphan labels')
    p.add_argument('--classes', default='classes.txt', help='classes file')
    args = p.parse_args()

    root = Path(args.root)
    orphan_dir = Path(args.orphan)
    classes_file = Path(args.classes)

    summary, orphan_list = fix_labels(root, orphan_dir, classes_file)
    out = []
    out.append('Label auto-fix summary')
    out.extend(summary)
    if orphan_list:
        out.append('\nSample moved orphan labels:')
        out.extend(orphan_list[:20])
    txt = '\n'.join(out)
    print(txt)


if __name__ == '__main__':
    main()
