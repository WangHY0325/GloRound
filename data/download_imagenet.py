import concurrent.futures as cf
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
REV = 'ac6a45918fb830d6d1555b01090d7b991e8f8a4a'
BASE = f'https://huggingface.co/datasets/benjamin-paine/imagenet-1k/resolve/{REV}/data/'
LOCAL = threading.local()
LOCK = threading.Lock()

def log(s):
    with LOCK:
        print(time.strftime('%H:%M:%S'), s, flush=True)

def session():
    if not hasattr(LOCAL, 's'):
        LOCAL.s = requests.Session()
        LOCAL.s.proxies = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
    return LOCAL.s

def get(url, headers=None):
    for attempt in range(8):
        try:
            r = session().get(url, headers=headers, timeout=(20, 120))
            r.raise_for_status()
            return r
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 20))

class RemoteFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        r = get(url + '?range=tail8', {'Range': 'bytes=-8'})
        if r.status_code != 206:
            raise RuntimeError('Source does not support range requests')
        self.size = int(r.headers['Content-Range'].split('/')[-1])
        self.pos = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        return self.pos
    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0: return b''
        start, end = self.pos, self.pos + n - 1
        r = get(self.url + f'?range={start}-{end}', {'Range': f'bytes={start}-{end}'})
        if r.status_code != 206 or len(r.content) != n:
            raise RuntimeError(f'Invalid range response: {start}-{end}')
        self.pos += n
        return r.content

def metadata(i):
    path = ROOT / 'metadata' / f'train-{i:05d}.json'
    if path.exists(): return json.loads(path.read_text())
    f = RemoteFile(BASE + f'train-{i:05d}-of-00310.parquet')
    p = pq.ParquetFile(f)
    result = {'shard': i, 'rows': p.metadata.num_rows}
    path.write_text(json.dumps(result))
    return result

def save_image(row, split, name, index):
    label = int(row['label'])
    wnid = CLASSES[str(label)][0]
    raw = row['image']['bytes']
    dest = ROOT / split / wnid / name
    digest = hashlib.sha256(raw).hexdigest()
    if not dest.exists():
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        temp = dest.with_suffix('.part')
        temp.write_bytes(raw)
        temp.replace(dest)
    elif hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'Existing image differs: {dest}')
    return {'index': int(index), 'label': label, 'wnid': wnid, 'file': dest.relative_to(ROOT).as_posix(), 'source_path': row['image'].get('path'), 'sha256': digest}

def train_shard(task):
    i, pairs = task
    manifest = ROOT / 'metadata' / f'train-extracted-{i:05d}.json'
    if manifest.exists(): return json.loads(manifest.read_text())
    f = RemoteFile(BASE + f'train-{i:05d}-of-00310.parquet')
    p = pq.ParquetFile(f)
    wanted = dict(pairs)
    start = 0
    records = []
    for group in range(p.metadata.num_row_groups):
        count = p.metadata.row_group(group).num_rows
        selected = [j for j in wanted if start <= j < start + count]
        if selected:
            table = p.read_row_group(group)
            for j in selected:
                row = table.slice(j-start, 1).to_pylist()[0]
                records.append(save_image(row, 'train', f'train_{wanted[j]:08d}.JPEG', wanted[j]))
        start += count
    manifest.write_text(json.dumps(records))
    return records

def validation_shard(i):
    manifest = ROOT / 'metadata' / f'val-extracted-{i:05d}.json'
    if manifest.exists(): return json.loads(manifest.read_text())
    path = ROOT / 'downloads' / f'validation-{i:05d}-of-00014.parquet'
    url = BASE + path.name
    if not path.exists():
        partial = path.with_suffix('.part')
        for attempt in range(8):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                with session().get(url + f'?download=1&offset={offset}', headers={'Range': f'bytes={offset}-'} if offset else {}, stream=True, timeout=(20,120)) as r:
                    r.raise_for_status()
                    if offset and r.status_code != 206:
                        offset = 0
                    with partial.open('ab' if offset else 'wb') as out:
                        for chunk in r.iter_content(4*1024*1024):
                            out.write(chunk)
                pq.ParquetFile(partial)
                partial.replace(path)
                break
            except Exception:
                if attempt == 7: raise
                time.sleep(min(2**attempt,20))
    records = []
    p = pq.ParquetFile(path)
    # HF shard splitting uses contiguous near-equal chunks.
    base = i * (50000 // 14) + min(i, 50000 % 14)
    for batch in p.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            idx = base + len(records)
            records.append(save_image(row, 'val', f'val_{idx:08d}.JPEG', idx))
    manifest.write_text(json.dumps(records))
    log(f'Validation shard {i+1}/14 extracted: {len(records)} images')
    return records

def main():
    global CLASSES
    for d in ['metadata', 'downloads']:
        (ROOT/d).mkdir(exist_ok=True)
    class_file = ROOT / 'imagenet_class_index.json'
    if not class_file.exists():
        r = get('https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
        class_file.write_bytes(r.content)
    CLASSES = json.loads(class_file.read_text())
    assert len(CLASSES) == 1000
    for split in ['train','val']:
        for wnid, _ in CLASSES.values():
            (ROOT/split/wnid).mkdir(parents=True,exist_ok=True)
    log('Reading training shard metadata')
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        meta = list(pool.map(metadata, range(310)))
    counts = [x['rows'] for x in meta]
    assert sum(counts) == 1281167
    indices = np.random.RandomState(3407).permutation(1281167)[:1024]
    (ROOT/'train_indices_seed3407.json').write_text(json.dumps(indices.tolist()))
    ends = np.cumsum(counts)
    tasks = {}
    for idx in indices:
        shard = int(np.searchsorted(ends, idx, side='right'))
        local = int(idx - (ends[shard-1] if shard else 0))
        tasks.setdefault(shard,[]).append((local,int(idx)))
    train_records, val_records = [], []
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        jobs = {pool.submit(validation_shard,i): ('val',i) for i in range(14)}
        jobs.update({pool.submit(train_shard,item): ('train',item[0]) for item in tasks.items()})
        done_train = 0
        for future in cf.as_completed(jobs):
            kind, i = jobs[future]
            records = future.result()
            if kind == 'val': val_records.extend(records)
            else:
                train_records.extend(records)
                done_train += 1
                if done_train % 10 == 0:
                    log(f'Training: {len(train_records)}/1024 images ({done_train}/{len(tasks)} shards)')
    by_idx = {r['index']:r for r in train_records}
    train_records = [by_idx[int(i)] for i in indices]
    val_records.sort(key=lambda r:r['index'])
    assert len(train_records)==1024 and len(val_records)==50000
    from collections import Counter
    assert set(Counter(r['label'] for r in val_records).values()) == {50}
    for split, records in [('train',train_records),('val',val_records),('calibration128',train_records[:128])]:
        (ROOT/f'{split}_manifest.json').write_text(json.dumps(records,indent=2))
    report = {'source':'benjamin-paine/imagenet-1k','revision':REV,'seed':3407,'train':1024,'val':50000,'image_bytes':'original parquet bytes, no resize or re-encoding','validation_class_counts':'1000 classes, 50 each','verification':'all images decoded/verified when extracted; SHA256 manifests recorded'}
    (ROOT/'download_report.json').write_text(json.dumps(report,indent=2))
    log('COMPLETE: 1024 training images and 50000 validation images verified')

if __name__ == '__main__':
    main()
