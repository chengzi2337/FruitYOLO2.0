import os
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
label_dirs = [ROOT / 'labels' / 'train', ROOT / 'labels' / 'val', ROOT / 'Dataset_Original' / 'labels' / 'train', ROOT / 'Dataset_Original' / 'labels' / 'val']
label_dirs = [d for d in label_dirs if d.exists()]

def scan_dir(d):
    counts = Counter()
    max_idx = -1
    min_idx = None
    sample_bad = []
    files = list(d.glob('*.txt'))
    for p in files:
        try:
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: 
                        continue
                    try:
                        idx = int(float(parts[0]))
                    except:
                        idx = parts[0]
                    counts[idx]+=1
                    if isinstance(idx, int):
                        if min_idx is None or idx < min_idx:
                            min_idx = idx
                        if idx > max_idx:
                            max_idx = idx
                        if idx > 50:
                            sample_bad.append((p.name, idx))
        except Exception:
            pass
    return {'files': len(files), 'counts': counts, 'min': min_idx, 'max': max_idx, 'sample_bad': sample_bad}

for d in label_dirs:
    res = scan_dir(d)
    print('DIR:', d)
    print('  label files:', res['files'])
    print('  index min/max:', res['min'], '/', res['max'])
    # show top indices
    most = res['counts'].most_common(20)
    print('  top indices:', most[:10])
    if res['sample_bad']:
        print('  sample bad entries (index>50):', res['sample_bad'][:5])
    print()
