import os
from pathlib import Path

root = Path('fresh-and-rotten-fruits-3') / 'test'
images_dir = root / 'images'
labels_dir = root / 'labels'
out_dir = root / 'labels_rewritten'
out_dir.mkdir(parents=True, exist_ok=True)

mapping = {
    'apple': {'healthy': 0, 'rotten': 5},
    'banana': {'healthy': 1, 'rotten': 3},
    'orange': {'healthy': 2, 'rotten': 4},
}

for lbl_path in labels_dir.glob('*.txt'):
    name = lbl_path.stem
    name_lower = name.lower()
    # determine fruit from filename
    if 'apple' in name_lower:
        fruit = 'apple'
    elif 'banana' in name_lower:
        fruit = 'banana'
    elif 'orange' in name_lower:
        fruit = 'orange'
    else:
        fruit = None

    out_lines = []
    with lbl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                orig_c = int(float(parts[0]))
            except:
                orig_c = None
            coords = parts[1:]

            # determine rotten/healthy from original class
            if orig_c is None:
                target_c = parts[0]
            else:
                is_rotten = orig_c >= 3
                if fruit is None:
                    # unknown fruit: keep original mapping
                    target_c = orig_c
                else:
                    kind = 'rotten' if is_rotten else 'healthy'
                    target_c = mapping[fruit][kind]

            out_lines.append(' '.join([str(target_c)] + coords))

    out_file = out_dir / lbl_path.name
    with out_file.open('w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))

print('Rewritten labels saved to', str(out_dir))
