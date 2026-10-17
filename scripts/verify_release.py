"""Verify release files, frozen membership, and optional prepared inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compare(left, right, path=''):
    import numpy as np
    if isinstance(right, dict):
        for key in right:
            compare(left[key], right[key], path + '/' + key)
    elif isinstance(right, list):
        if right and isinstance(right[0], (int, float)):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, err_msg=path)
        elif left != right:
            raise AssertionError('Different values: ' + path)
    elif isinstance(right, (int, float)):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, err_msg=path)
    elif left != right:
        raise AssertionError('Different values: ' + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', action='store_true')
    args = parser.parse_args()
    filelist = ROOT / 'MANIFEST.sha256'
    if not filelist.exists():
        raise FileNotFoundError('Release manifest not found')
    checked = 0
    for line in filelist.read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or digest(path) != expected:
            raise ValueError('Release file changed: ' + relative)
        checked += 1
    split_dir = ROOT / 'splits/formal_pub_527'
    split_manifest = json.loads((split_dir / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in split_manifest['sha256'].items():
        if digest(split_dir / name) != expected:
            raise ValueError('Frozen split file changed: ' + name)
    ids = {r: (split_dir / (r + '.txt')).read_text().split() for r in ('train', 'val', 'test')}
    if {r: len(v) for r, v in ids.items()} != {'train': 370, 'val': 50, 'test': 107} or len(set(sum(ids.values(), []))) != 527:
        raise ValueError('Invalid frozen split')
    if args.cache:
        sys.path.insert(0, str(ROOT / 'src'))
        from training import load_protocol
        import numpy as np
        _, meta = load_protocol()
        compare(meta, json.loads((ROOT / 'reference_results/cache_metadata.json').read_text(encoding='utf-8')))
        compare(json.loads((ROOT / 'data/formal/scaler.json').read_text(encoding='utf-8')),
                json.loads((ROOT / 'reference_results/preprocessing.json').read_text(encoding='utf-8')))
        with np.load(ROOT / 'data/formal/formal.npz', allow_pickle=False) as cache:
            if set(k for k in cache.files if k.startswith('q_')) != {'q_train', 'q_val'}:
                raise ValueError('Unexpected labels in cache')
            if cache['x_dynamic'].shape[2] != 5 or cache['x_static'].shape != (527, 21):
                raise ValueError('Unexpected feature dimensions')
    print(json.dumps({'status': 'passed', 'release_files': checked,
                      'basins': 527, 'cache_checked': args.cache}))


if __name__ == '__main__':
    main()
