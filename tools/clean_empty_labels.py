import os
import argparse

def is_empty_label(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for ln in f:
                if ln.strip():
                    return False
    except Exception:
        return True
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True)
    parser.add_argument('--images', required=True)
    args = parser.parse_args()

    labels_dir = args.labels
    images_dir = args.images
    if not os.path.isdir(labels_dir):
        print('Labels dir not found:', labels_dir); return 1
    if not os.path.isdir(images_dir):
        print('Images dir not found:', images_dir); return 1

    txts = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
    removed = []
    for t in txts:
        p = os.path.join(labels_dir, t)
        if is_empty_label(p):
            # remove label and corresponding image(s)
            try:
                os.remove(p)
            except Exception as e:
                print('Failed remove label', p, e)
            base = os.path.splitext(t)[0]
            # possible image extensions
            for ext in ('.jpg','.jpeg','.png'):
                img = os.path.join(images_dir, base + ext)
                if os.path.exists(img):
                    try:
                        os.remove(img)
                    except Exception as e:
                        print('Failed remove image', img, e)
            removed.append(t)

    # also remove images without any label file
    imgs = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    for im in imgs:
        base = os.path.splitext(im)[0]
        lbl = os.path.join(labels_dir, base + '.txt')
        if not os.path.exists(lbl):
            # remove image
            try:
                os.remove(os.path.join(images_dir, im))
                removed.append(base + ' (no label)')
            except Exception as e:
                print('Failed remove image without label', im, e)

    print('Removed', len(removed), 'items. Examples:', removed[:20])
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
