import os
import shutil
import argparse
import json

"""
Remap external dataset labels to this project's class ordering.

Usage examples (after you copy external dataset into workspace):
python tools/remap_external_labels.py \
  --src external_test \
  --dst_images Dataset_Original/images/test \
  --dst_labels Dataset_Original/labels/test \
  --src_classes external_test/classes.txt

If the source has a classes.txt, the script will read it and build a mapping
by matching class names. If names don't match, provide a JSON mapping file
with keys being source class names and values being target class names, e.g.
  {"apple_fresh":"Apple_healthy", "apple_rotten":"Apple_rotten", ...}

The script copies images and rewrites YOLO txt labels (class indices) using
the target ordering from this project's `data.yaml`.
"""

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_target_names():
    import yaml
    root = os.path.dirname(os.path.dirname(__file__))
    # try repo-root data.yaml
    p = os.path.join(root, 'data.yaml')
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        names = d.get('names')
        if names:
            return names
    # try repo-root classes.txt
    ct = os.path.join(root, 'classes.txt')
    if os.path.exists(ct):
        with open(ct, 'r', encoding='utf-8') as f:
            return [ln.strip() for ln in f if ln.strip()]
    # try current working directory as fallback
    p2 = os.path.join(os.getcwd(), 'data.yaml')
    if os.path.exists(p2):
        with open(p2, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        names = d.get('names')
        if names:
            return names
    ct2 = os.path.join(os.getcwd(), 'classes.txt')
    if os.path.exists(ct2):
        with open(ct2, 'r', encoding='utf-8') as f:
            return [ln.strip() for ln in f if ln.strip()]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Source dataset root (images and labels subfolders)')
    parser.add_argument('--src_classes', help='Source classes.txt path (optional)')
    parser.add_argument('--mapping', help='JSON file mapping source class name -> target class name')
    parser.add_argument('--dst_images', default='Dataset_Original/images/test', help='Destination images folder')
    parser.add_argument('--dst_labels', default='Dataset_Original/labels/test', help='Destination labels folder')
    args = parser.parse_args()

    src = args.src
    if not os.path.exists(src):
        print('Source path not found:', src)
        return

    src_images = os.path.join(src, 'images') if os.path.isdir(os.path.join(src,'images')) else src
    src_labels = os.path.join(src, 'labels') if os.path.isdir(os.path.join(src,'labels')) else os.path.join(src, 'labels')

    dst_images = args.dst_images
    dst_labels = args.dst_labels
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    target_names = load_target_names()
    if target_names is None:
        print('Could not load target names from data.yaml')
        return

    # load source names
    src_names = []
    if args.src_classes and os.path.exists(args.src_classes):
        src_names = read_lines(args.src_classes)
    else:
        # try common files
        for fname in ['classes.txt', 'names.txt']:
            p = os.path.join(src, fname)
            if os.path.exists(p):
                src_names = read_lines(p)
                break

    mapping = {}
    if args.mapping:
        with open(args.mapping, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

    # build source->target index map
    src_to_target_idx = {}
    if src_names:
        for i, sname in enumerate(src_names):
            # if mapping provided map by name
            tgt_name = mapping.get(sname) if mapping else None
            if not tgt_name:
                # try exact match ignoring case and non-alnum
                for tn in target_names:
                    if tn.lower() == sname.lower():
                        tgt_name = tn
                        break
            if not tgt_name:
                print(f'Warning: source class "{sname}" not mapped to any target; set to -1')
                src_to_target_idx[i] = -1
            else:
                try:
                    tidx = target_names.index(tgt_name)
                    src_to_target_idx[i] = tidx
                except ValueError:
                    print(f'Warning: mapped target name "{tgt_name}" not found in target names')
                    src_to_target_idx[i] = -1
    elif mapping:
        # mapping keys are source names
        # invert target names to idx
        t2i = {n: i for i,n in enumerate(target_names)}
        for sname, tgt_name in mapping.items():
            src_to_target_idx[sname] = t2i.get(tgt_name, -1)
    else:
        print('No source class list or mapping provided. Cannot remap automatically.')
        return

    # copy images and remap labels
    imgs = [f for f in os.listdir(src_images) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    print('Found', len(imgs), 'images in', src_images)

    for im in imgs:
        src_im = os.path.join(src_images, im)
        dst_im = os.path.join(dst_images, im)
        shutil.copy2(src_im, dst_im)
        base = os.path.splitext(im)[0]
        src_lbl = os.path.join(src_labels, base + '.txt')
        dst_lbl = os.path.join(dst_labels, base + '.txt')
        if not os.path.exists(src_lbl):
            # create empty label
            open(dst_lbl, 'w', encoding='utf-8').close()
            continue
        out_lines = []
        with open(src_lbl, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                s_idx = parts[0]
                try:
                    s_idx_i = int(s_idx)
                except Exception:
                    # maybe label file uses names; try mapping
                    if s_idx in src_names:
                        s_idx_i = src_names.index(s_idx)
                    else:
                        print('Unrecognized label token:', s_idx, 'in', src_lbl)
                        continue
                tgt_idx = src_to_target_idx.get(s_idx_i, -1)
                if tgt_idx < 0:
                    # skip or set to 0; here we skip
                    continue
                out_lines.append(' '.join([str(tgt_idx)] + parts[1:]))
        with open(dst_lbl, 'w', encoding='utf-8') as fw:
            fw.write('\n'.join(out_lines))

    print('Remapping complete. Images copied to', dst_images, 'labels to', dst_labels)


if __name__ == '__main__':
    main()
